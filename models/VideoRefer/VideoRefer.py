import math
import os
import argparse
import json
import warnings
from tqdm import tqdm
import transformers
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List
import torch
import sys
sys.path.append('models/VideoRefer/VideoRefer')
from videorefer import model_init, mm_infer
from pycocotools import mask as maskUtils
import numpy as np
from videorefer.mm_utils import process_video
from functools import partial
from matplotlib import pyplot as plt
from PIL import Image
from videorefer.utils import disable_torch_init        
from decord import VideoReader, cpu
import cv2
import re 
from ..utils import load_video

def truncate_after_first_period(input_string):
    period_index = input_string.find('.')
    
    if period_index != -1 and period_index < len(input_string) - 1:
        return input_string[period_index + 1:]
    else:
        return ''

class VideoRefer:
    def __init__(self,model_path):
        super().__init__()
        self.model, self.processor, self.tokenizer = model_init(model_path)
        for m in self.model.modules():
            m.tokenizer = self.tokenizer
        
    def generate_output(self,messages):
        video = messages["video"]
        prompt = messages["prompt"]
        box = messages["box"]
        fps = messages["fps"]

        prefix = f'There are {len(box)} objects in the video, '
        for i in range(len(box)):
            prefix+=f'<object {i}>[<region>], '
        prefix = prefix[:-2]
        prompt = prefix+'. '+truncate_after_first_period(prompt)
        video_frames, _ = load_video(video, fps=fps, num_segments=16, add_last_frame=False)
        # video_frames, _ = load_video(video, fps=fps, num_segments=16)
        video_tensor, frame_tensor, height, width = process_video(video_frames, processor=self.processor, aspect_ratio='square', num_frames=16, frame_idx=[len(video_frames)-1])
        masks = []
        for b in box:
            b = [max(0,b_) for b_ in b]
            h, w = height, width
            masks_ = np.zeros((h, w))
            masks_[b[1]: b[3], b[0]: b[2]] = 1
            masks.append(masks_)
            
        masks_ = np.array(masks)
        masks_tensor = torch.Tensor(masks_)
        masks_tensor = masks_tensor.unsqueeze(0)

        frame_nums = [1]
        ann_indices = [[[0]]*len(box)]

        output = mm_infer(
                video_tensor,
                prompt,
                model=self.model,
                tokenizer=self.tokenizer,
                masks=masks_tensor.cuda(),
                frame=frame_tensor,
                ann_indices=ann_indices,
                frame_nums=frame_nums,
            )
        return output
    
    def generate_outputs(self,messages_list):
        res = []
        for messages in messages_list:
            result = self.generate_output(messages)
            res.append(result)
        return res
    