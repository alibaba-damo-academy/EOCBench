from PIL import Image
import numpy as np
import torch
from transformers import VideoLlavaProcessor, VideoLlavaForConditionalGeneration

from ..utils import load_video

class VideoLLava:
    def __init__(self,model_path):
        super().__init__()
        self.llm = VideoLlavaForConditionalGeneration.from_pretrained(model_path,torch_dtype=torch.bfloat16, device_map="cuda",attn_implementation="flash_attention_2")
        self.processor = VideoLlavaProcessor.from_pretrained(model_path)

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

        prompt = f"USER: <video>{prompt}? ASSISTANT:"

        video_inputs = [Image.open(frame) for frame in video]
        # 转换为numpy
        video_inputs = [np.array(frame) for frame in video_inputs]
        video_inputs = np.stack(video_inputs)


        llm_inputs = {
            "prompt": prompt,
            "video": video_inputs
        }

        return llm_inputs


    def generate_output(self,messages):
        llm_inputs = self.process_messages(messages)
        prompt = llm_inputs["prompt"]
        video = llm_inputs["video"]
        inputs = self.processor(text=prompt, videos=video, return_tensors="pt",max_length=8192,truncation=True)
        inputs = inputs.to("cuda")
        generate_ids = self.llm.generate(**inputs,do_sample = False,temperature=None)
        outputs = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        prompt = prompt.replace("<video>", "")
        outputs = outputs.replace(prompt, "").strip()
        return outputs
    
    def generate_outputs(self,messages_list):
        res = []
        for messages in messages_list:
            result = self.generate_output(messages)
            res.append(result)
        return res