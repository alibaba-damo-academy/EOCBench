from longva.model.builder import load_pretrained_model
from longva.mm_utils import tokenizer_image_token, process_images
from longva.constants import IMAGE_TOKEN_INDEX
from PIL import Image
from decord import VideoReader, cpu
import torch
import numpy as np

from ..utils import load_video

class LongVA:
    def __init__(self,model_path):
        super().__init__()
        self.gen_kwargs = {"do_sample": False, "temperature": None, "top_p": None, "num_beams": 1, "use_cache": True, "max_new_tokens": 1024}
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(model_path, None, "llava_qwen", device_map="cuda")


    
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

        video,frame_times = load_video(video_path,fps, num_segments=num_segments, fps_segments=fps_segments)

        video_inputs = video
        # print(len(video_inputs))
        # 转换为numpy
        video_inputs = [np.array(frame) for frame in video_inputs]
        video_inputs = np.stack(video_inputs)
        video_inputs = self.image_processor.preprocess(video_inputs, return_tensors="pt")["pixel_values"].to(self.model.device, dtype=torch.float16)

        question = prompt
        user_prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<image>\n{question}<|im_end|>\n<|im_start|>assistant\n"
        input_ids = tokenizer_image_token(user_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.model.device)

        llm_inputs = {
            "input_ids": input_ids,
            "video": video_inputs
        }

        return llm_inputs


    def generate_output(self,messages):
        llm_inputs = self.process_messages(messages)
        video = llm_inputs["video"]
        input_ids = llm_inputs["input_ids"]
        cont = self.model.generate(
            input_ids,
            images=[video],
            modalities= ["video"],
            **self.gen_kwargs
        )
        text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
        return text_outputs
    
    def generate_outputs(self,messages_list):
        res = []
        for messages in messages_list:
            result = self.generate_output(messages)
            res.append(result)
        return res
