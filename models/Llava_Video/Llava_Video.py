from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from PIL import Image
import copy
import warnings
import os
from decord import VideoReader, cpu
import numpy as np

from ..utils import load_video


class LlavaVideo:
    def __init__(self,model_path):
        super().__init__()
        model_name = "llava_qwen"
        self.device = "cuda"
        device_map = "auto"
        self.tokenizer, self.llm, self.image_processor, self.max_length = load_pretrained_model(model_path, None, model_name, torch_dtype="bfloat16", device_map=device_map)
        self.llm.eval()
    
    def process_messages(self,messages):
        prompt = messages["prompt"]
        video_path = messages["video"]   
        frame_type = messages["frame_type"]
        fps = messages["fps"]
        if "frames" in frame_type:
            num_segments = int(frame_type.split("_")[0])
            fps_segments = None
        elif "fps" in frame_type:
            num_segments = None
            fps_segments = int(frame_type.split("_")[0])
        else:
            raise "only support frames"

        video, frame_times = load_video(video_path,fps, num_segments=num_segments, fps_segments=fps_segments)

        video_prefix = ''.join([f'{frame_time}s: <image>\n' for frame_time in frame_times])

        video_inputs = video
        # 转换为numpy
        video_inputs = [np.array(frame) for frame in video_inputs]
        video_inputs = np.stack(video_inputs)
        video_inputs = self.image_processor.preprocess(video_inputs, return_tensors="pt")["pixel_values"].cuda().bfloat16()
        
        conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
        question = video_prefix + prompt
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)

        llm_inputs = {
            "input_ids": input_ids,
            "video": video_inputs
        }

        return llm_inputs


    def generate_output(self,messages):
        llm_inputs = self.process_messages(messages)
        video = llm_inputs["video"]
        input_ids = llm_inputs["input_ids"]
        cont = self.llm.generate(
            input_ids,
            images=video,
            modalities= ["video"],
            do_sample=False,
            temperature=0,
            max_new_tokens=4096,
        )
        text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
        return text_outputs
    
    def generate_outputs(self,messages_list):
        res = []
        for messages in messages_list:
            result = self.generate_output(messages)
            res.append(result)
        return res

