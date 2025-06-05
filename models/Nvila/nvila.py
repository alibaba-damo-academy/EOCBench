import os
from collections import defaultdict

import cv2
import glob
import numpy as np
import torch
from PIL import Image
from transformers import AutoConfig, AutoModel, PretrainedConfig
import decord
from decord import VideoReader, cpu
try:
    import llava
    import llava.model
except:
    llava = None
from ..utils import load_video

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<vila/video>"


def process_image(
    image_file, data_args, image_folder, enable_dynamic_res=False, enable_dynamic_s2=False, max_tiles=None
):
    processor = data_args.image_processor
    if isinstance(image_file, str):
        if image_folder is not None:
            image = Image.open(os.path.join(image_folder, image_file)).convert("RGB")
        else:
            image = Image.open(image_file).convert("RGB")
    else:
        # image is stored in bytearray
        image = image_file
    image = image.convert("RGB")
    if hasattr(data_args.image_processor, "crop_size"):
        # CLIP vision tower
        crop_size = data_args.image_processor.crop_size
    else:
        # SIGLIP vision tower
        assert hasattr(data_args.image_processor, "size")
        crop_size = data_args.image_processor.size
    if "dynamic_s2" in data_args.image_aspect_ratio and enable_dynamic_s2:
        assert crop_size["height"] == crop_size["width"]
        images, block_size = dynamic_s2_preprocess(
            image, s2_scales=data_args.s2_scales, max_num=data_args.max_tiles, image_size=crop_size["height"]
        )
        images = [processor.preprocess(image, return_tensors="pt")["pixel_values"][0] for image in images]
        return torch.stack(images), block_size
    if "dynamic" in data_args.image_aspect_ratio and enable_dynamic_res:
        assert crop_size["height"] == crop_size["width"]
        if max_tiles is not None:
            max_num = max_tiles
        else:
            max_num = data_args.max_tiles
        images = dynamic_preprocess(image, min_num=data_args.min_tiles, max_num=max_num, image_size=crop_size["height"])
        images = [processor.preprocess(image, return_tensors="pt")["pixel_values"][0] for image in images]
        return torch.stack(images)

    if data_args.image_aspect_ratio == "resize":
        image = image.resize((crop_size["width"], crop_size["height"]))
    if data_args.image_aspect_ratio == "pad":

        def expand2square(pil_img, background_color):
            width, height = pil_img.size
            if width == height:
                return pil_img
            elif width > height:
                result = Image.new(pil_img.mode, (width, width), background_color)
                result.paste(pil_img, (0, (width - height) // 2))
                return result
            else:
                result = Image.new(pil_img.mode, (height, height), background_color)
                result.paste(pil_img, ((height - width) // 2, 0))
                return result

        image = expand2square(image, tuple(int(x * 255) for x in processor.image_mean))
        image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
    else:
        # Using default behavior of the vision encoder
        # For CLIP, default is central crop
        # For Radio, default is central crop
        # For Siglip, default is resize
        # For InternVIT, default is resize
        image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
    return image


def get_index(bound, fps, max_frame, first_idx=0, num_segments=32,fps_segments=None):
    frame_ids = []
    max_frame = max_frame + 1

    if num_segments==1:
        return max_frame
    if not fps_segments:
        n_frames = max_frame // (num_segments - 1)
    else:
        n_frames = fps//fps_segments
    if n_frames == 0:
        return list(range(max_frame))
    for frame_count in range(max_frame):
        if (frame_count % n_frames == 0 or frame_count == max_frame) and len(frame_ids) < num_segments:
            frame_ids.append(frame_count)
    return frame_ids

class NVILA:

    def __init__(
        self,
        model_path
    ):
        self.model = llava.load(
            model_path,
            # torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2",
            # **kwargs
        )
        self.tokenizer = self.model.tokenizer
        self.model_cfg = self.model.config
        self.model_cfg.image_processor = self.model.vision_tower.image_processor


    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
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

        video,frame_times = load_video(video_path,fps, num_segments=num_segments,fps_segments)
        
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
        
        conversation = []
        for message in messages:
            if isinstance(message["content"], list):
                text = ""
                for content in message["content"]:
                    if content["type"] == "text":
                        text += content["text"]
                    elif content["type"] == "image":
                        text += DEFAULT_IMAGE_TOKEN
                    elif content["type"] == "video":
                        text += DEFAULT_VIDEO_TOKEN
                    else:
                        raise ValueError(f"Unsupported type: {content['type']}")
                conversation.append({"role": message["role"], "content": text.strip()})
            elif isinstance(message["content"], str):
                conversation.append({"role": message["role"], "content": message["content"].strip()})
            else:
                raise ValueError(f"Unsupported type: {type(message['content'])}")
        prompt = self.tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=add_generation_prompt,
            tokenize=tokenize,
            **kwargs,
        )
        return prompt, video
    
    def generate_outputs(self,messages_list):
        res = []
        for messages in messages_list:
            result = self.generate_output(messages)
            res.append(result)
        return res
    
    def generate_output(self,messages):
        # print(messages)
        prompt, images = self.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = self.process_text(prompt)
        image_inputs = self.process_image(images, return_tensors="pt")
        
        input_ids = input_ids['input_ids'].cuda()
        modal = 'video'
        media = {modal: [image_inputs.cuda().half()]}

        do_sample = False 
        temperature = 0.2 if do_sample else 0.0 
        top_p = 0.9 
        max_new_tokens = 2048
        # print(input_ids)
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids=input_ids,
                media=media,
                media_config = defaultdict(dict),
                do_sample=do_sample,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                top_p=top_p,
            )

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return outputs
        
        

    def process_text(self, text, *args, **kwargs):
        return self.tokenizer(text, return_tensors="pt", **kwargs)

    def process_image(self, images, **kwargs):
        new_images = [process_image(image, self.model_cfg, None, enable_dynamic_res=False) for image in images]
        if all(x.shape == new_images[0].shape for x in new_images):
            if len(new_images[0].shape) == 4:
                new_images = torch.cat(new_images, dim=0)
            elif len(new_images[0].shape) == 3:
                new_images = torch.stack(new_images, dim=0)
            else:
                raise ValueError(f"new_images rank does not equal to 4, rank: {len(new_images[0].shape)}")
        else:
            raise ValueError("The shape of images in new_images is different!")
        return new_images


def model_init(model_path="Efficient-Large-Model/NVILA-8B", device_map="auto", **kwargs):
    if llava is None:
        raise ImportError("Please install NVILA following https://github.com/NVlabs/VILA/tree/main")
    model = llava.load(
        model_path,
        # torch_dtype=torch.bfloat16,
        device_map=device_map,
        attn_implementation="flash_attention_2",
        **kwargs
    )
    processor = NVILAProcessor(
        tokenizer=model.tokenizer,
        image_processor=model.vision_tower.image_processor,
        model_cfg=model.config,
    )
    return model, processor


def mm_infer(data_dict, model, tokenizer, modal='video', **kwargs):
    input_ids = data_dict["input_ids"].cuda()
    media = {modal: [data_dict["images"].cuda().half()]}

    do_sample = kwargs.get('do_sample', False)
    temperature = kwargs.get('temperature', 0.2 if do_sample else 0.0)
    top_p = kwargs.get('top_p', 0.9)
    max_new_tokens = kwargs.get('max_new_tokens', 2048)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=input_ids,
            media=media,
            media_config = defaultdict(dict),
            do_sample=do_sample,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return outputs
