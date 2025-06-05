from SPHINX_V import SPHINX_V_Model
from PIL import Image
import os
import torch
import torch.distributed as dist
import pdb
from ..utils import get_second_last_frame_from_video

def truncate_after_first_period(input_string):
    period_index = input_string.find('.')
    
    if period_index != -1 and period_index < len(input_string) - 1:
        return input_string[period_index + 1:]
    else:
        return ''

class SPHINX_V():
    def __init__(self, model_path):
        self.model = SPHINX_V_Model.from_pretrained(
            pretrained_path=model_path, 
            llama_type="llama_ens5_vp",
            llama_config="/mnt/workspace/workgroup/yuanyq/checkpoints/SPHINX-V-Model/llama-2-13b/params.json",
            with_visual=True,
            mp_group=None # dist.new_group(ranks=list(range(world_size)))
        )
    
    def generate_output(self,messages):
        video = messages["video"]
        prompt = messages["prompt"]
        image = get_second_last_frame_from_video(video)
        image = Image.fromarray(image).convert('RGB')
        box = messages["box"]
        qas = [[prompt, None]]
        vps = box
        
        response = self.model.generate_response(qas, vps, image, max_gen_len=1024, temperature=0.9, top_p=0.5, seed=0)
        return response
        
    def generate_outputs(self,messages_list):
        res = []
        for messages in messages_list:
            result = self.generate_output(messages)
            res.append(result)
        # print(res)
        return res
    