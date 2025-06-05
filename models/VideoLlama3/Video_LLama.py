import torch
import os
import numpy as np
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image


from decord import VideoReader, cpu
import torch
import numpy as np
import cv2


from ..utils import load_video

class VideoLLama:
    def __init__(self,model_path):
        super().__init__()
        device = "cuda"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map={"": device},
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

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

        video,frame_times = load_video(video_path, fps, num_segments=num_segments, fps_segments=fps_segments)
        video_inputs = [np.array(frame) for frame in video]
        messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video",
                                "video": video_inputs,
                                "timestamps" : frame_times,
                                "num_frames": len(frame_times)
                            },
                            {"type": "text", "text": prompt}
                        ],
                    }
                ]
        inputs = self.processor(
            conversation=messages,
            add_system_prompt=True,
            add_generation_prompt=True,
            return_tensors="pt"
        )
        inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        # import pdb;pdb.set_trace()
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        return inputs


    def generate_output(self,messages):
        inputs = self.process_messages(messages)
        output_ids = self.model.generate(**inputs, max_new_tokens=128)
        response = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return response
    
    def generate_outputs(self,messages_list):
        res = []
        for messages in messages_list:
            result = self.generate_output(messages)
            res.append(result)
        # print(res)
        return res
