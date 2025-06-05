import argparse
import torch
import os
import json
from tqdm import tqdm
from functools import partial
from transformers import AutoTokenizer, CLIPImageProcessor
from osprey.constants import IMAGE_TOKEN_INDEX
from osprey.conversation import conv_templates, SeparatorStyle
from osprey.mm_utils import tokenizer_image_token
from osprey.train.train import preprocess_multimodal
from osprey.train.train import DataArguments
from osprey.model.language_model.osprey_llama import OspreyLlamaForCausalLM
from pycocotools import mask as maskUtils
import numpy as np
from PIL import Image
import argparse
import cv2
from ..utils import get_second_last_frame_from_video

data_args = DataArguments()
data_args.mm_use_im_start_end = False
data_args.is_multimodal = True

def truncate_after_first_period(input_string):
    period_index = input_string.find('.')
    
    if period_index != -1 and period_index < len(input_string) - 1:
        return input_string[period_index + 1:]
    else:
        return ''

class Osprey():
    def __init__(self, model_path):

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            model_max_length=2048,
            padding_side="right",
            use_fast=True
        )
        self.model = OspreyLlamaForCausalLM.from_pretrained(
                                                model_path,
                                                torch_dtype=torch.bfloat16,
                                                ).cuda()
        self.tokenizer.pad_token = self.tokenizer.unk_token

        self.image_processor = CLIPImageProcessor(do_resize=True, size={"shortest_edge":512}, resample=3,  do_center_crop=True, crop_size={"height": 512, "width": 512},
                                                  do_rescale=True, rescale_factor=0.00392156862745098, do_normalize=True, image_mean=[0.48145466, 0.4578275, 0.40821073],
                                                  image_std=[0.26862954, 0.26130258, 0.27577711], do_convert_rgb=True, )
        
        spi_tokens = ['<mask>', '<pos>']
        self.tokenizer.add_tokens(spi_tokens, special_tokens=True)
        
        for m in self.model.modules():
            m.tokenizer = self.tokenizer

        vision_tower = self.model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        vision_tower.to(dtype=torch.float16, device='cuda')
        

    def generate_output(self, messages):
        img_path = messages["video"][-1]
        prompt = messages["prompt"]
        box = messages["box"]
        
        prefix = f'There are {len(box)} objects in the video, '
        for i in range(len(box)):
            prefix+=f'<object {i}><mask><pos>, '
        prefix = prefix[:-2]
        prompt = prefix+'. '+truncate_after_first_period(prompt)
        
        image = get_second_last_frame_from_video(messages["video"])
        masks = []
        for b in box:
            b = [max(0,b_) for b_ in b]
            h, w = image.shape[:2]
            masks_ = np.zeros((h, w))
            masks_[b[1]: b[3], b[0]: b[2]] = 1
            masks.append(masks_)
        masks = torch.tensor(masks)
 
        init_inputs = get_init_inputs(image,
                                    self.image_processor,
                                    mask=masks,
                                    prompt=prompt
                                    )

        image = init_inputs['image']
        masks = init_inputs['masks'].cuda()

        conv = conv_templates['osprey_v1'].copy()
        qs = init_inputs['sources'][0][0]['value']

        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

        self.model.model.tokenizer = self.tokenizer

        with torch.inference_mode():

            self.model.orig_forward = self.model.forward
            self.model.forward = partial(self.model.orig_forward,
                                        img_metas=[None],
                                        masks=[masks.half()])

            output_ids = self.model.generate(
                input_ids,
                images=image.unsqueeze(0).half().cuda(),
                do_sample=True,
                temperature=0.2,
                max_new_tokens=1024,
                use_cache=True,
                num_beams=1,
            )

            self.model.forward = self.model.orig_forward

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (
            input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(
                f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:],
                                            skip_special_tokens=True)[0]
        return outputs

    def generate_outputs(self,messages_list):
        res = []
        for messages in messages_list:
            result = self.generate_output(messages)
            res.append(result)
        return res
    
def get_init_inputs(image,
                    processor,
                    mask,
                    prompt):

       
    image = Image.fromarray(image).convert('RGB')

    image = processor.preprocess(image,
                                    do_center_crop=False,
                                    return_tensors='pt')['pixel_values'][0]

    image = torch.nn.functional.interpolate(image.unsqueeze(0),
                                            size=(512, 512),
                                            mode='bilinear',
                                            align_corners=False).squeeze(0)


    cur_token_len = (image.shape[1] // 16) * (image.shape[2] // 16)

    mask = mask.to(image.device)

    begin_str = """<image>\nThis provides an overview of the picture.\n"""

    sources = dict()
    sources['conversations'] = []

    sources['conversations'].append({'from': 'human', 'value': begin_str+prompt})
    
    sources = preprocess_multimodal([sources['conversations']], data_args, cur_token_len)

    data_dict = {}
    data_dict['sources'] = sources
    data_dict['image'] = image
    data_dict['masks'] = mask
    return data_dict


