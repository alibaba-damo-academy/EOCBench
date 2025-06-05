from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor,Qwen2_5_VLProcessor
from qwen_vl_utils import process_vision_info

import os

from ..utils import load_video

class Qwen2_5_VL:
    def __init__(self,model_path):
        super().__init__()
        self.llm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype="auto", device_map="auto",attn_implementation="flash_attention_2")
        self.processor = AutoProcessor.from_pretrained(model_path)


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
        messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video",
                                "video": video,

                            },
                            {"type": "text", "text": prompt}
                        ],
                    }
                ]
        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        return inputs

    def generate_output(self,messages):
        inputs = self.process_messages(messages)
        generated_ids = self.llm.generate(**inputs,do_sample=False,repetition_penalty=1,max_new_tokens=1024)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0]
    
    def generate_outputs(self,messages_list):
        res = []
        for messages in messages_list:
            result = self.generate_output(messages)
            res.append(result)
        return res
