import argparse
import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer
from llava.visual_prompt_generator import image_blending
import cv2
from ..utils import load_last_frame

def truncate_after_first_period(input_string):
    period_index = input_string.find('.')
    
    if period_index != -1 and period_index < len(input_string) - 1:
        return input_string[period_index + 1:]
    else:
        return ''

def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


class VIP_LLava():
    def __init__(self, model_path):
        model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, context_len = load_pretrained_model(model_path, None, model_name, device='cuda')
        self.model_name = model_name
    
    def generate_output(self, messages):
        model_name = self.model_name
        if 'llama-2' in model_name.lower():
            conv_mode = "llava_llama_2"
        elif "v1" in model_name.lower() or 'vip-llava' in model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"

        conv = conv_templates[conv_mode].copy()
        if "mpt" in model_name.lower():
            roles = ('user', 'assistant')
        else:
            roles = conv.roles
            
        video = messages["video"]
        inp = truncate_after_first_period(messages["prompt"])
        
        image = load_last_frame(video)
        image = Image.fromarray(image)
        
        image_tensor = process_images([image], self.image_processor, self.model.config)
        if type(image_tensor) is list:
            image_tensor = [image.to(self.model.device, dtype=torch.float16) for image in image_tensor]
        else:
            image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)
        
        if self.model.config.mm_use_im_start_end:
            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
        else:
            inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
        conv.append_message(conv.roles[0], inp)
        
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.model.device)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        temperature = 0.2
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True if temperature > 0 else False,
                temperature=temperature,
                max_new_tokens=512,
                streamer=streamer,
                use_cache=True,
                stopping_criteria=[stopping_criteria])
        outputs = self.tokenizer.batch_decode(output_ids)[0].strip()
        return outputs


    def generate_outputs(self,messages_list):
        print(messages_list)
        res = []
        for messages in messages_list:
            result = self.generate_output(messages)
            res.append(result)
        return res
    
