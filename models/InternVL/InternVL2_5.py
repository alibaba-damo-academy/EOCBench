import os
import torch
from transformers import AutoModel, AutoTokenizer
import math

from .utils import load_image,load_video

def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers = {
        'InternVL2_5-1B': 24, 'InternVL2_5-2B': 24, 'InternVL2_5-4B': 36, 'InternVL2_5-8B': 32,
        'InternVL2_5-26B': 48, 'InternVL2_5-38B': 64, 'InternVL2_5-78B': 80}[model_name]
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.model.rotary_emb'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0
    return device_map

class InternVL:
    def __init__(self,model_path):
        super().__init__()
        self.llm =  AutoModel.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                    device_map= "auto",
                    attn_implementation="flash_attention_2"
                    ).eval()

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        self.generation_config = dict(max_new_tokens=1024, do_sample=False,pad_token_id=self.tokenizer.pad_token_id)
    
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

        pixel_values, num_patches_list,frame_times = load_video(video_path, fps=fps, num_segments=num_segments, max_num=1,fps_segments=fps_segments)
        pixel_values = pixel_values.to(torch.bfloat16).cuda()
        video_prefix = ''.join([f'{frame_time}s: <image>\n' for frame_time in frame_times])

        question = video_prefix + prompt
        
        pixel_values = pixel_values.to("cuda")
        llm_inputs = {
            "question": question,
            "pixel_values": pixel_values,
            "num_patches_list": num_patches_list,
        }

        return llm_inputs


    def generate_output(self,messages):
        llm_inputs = self.process_messages(messages)
        question = llm_inputs["question"]
        pixel_values = llm_inputs["pixel_values"]
        num_patches_list = llm_inputs["num_patches_list"]
        response, history = self.llm.chat(self.tokenizer, pixel_values, question, self.generation_config,
                               num_patches_list=num_patches_list,
                               history=None, return_history=True)
        return response
    
    def generate_outputs(self,messages_list):
        res = []
        for messages in messages_list:
            result = self.generate_output(messages)
            res.append(result)
        return res

