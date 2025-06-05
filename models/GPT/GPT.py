from abc import abstractmethod
from tqdm.asyncio import tqdm_asyncio 
import os
import logging
import asyncio
import aiohttp
import requests
import base64
import numpy as np
from tenacity import (
    retry,
    stop_after_attempt,
    wait_fixed,
)
from tqdm import tqdm
from PIL import Image
from decord import VideoReader, cpu
from io import BytesIO
from openai import AzureOpenAI, OpenAI,AsyncAzureOpenAI,AsyncOpenAI

from ..utils import load_video 


def encode_image(image_path):
    if isinstance(image_path,str):
        image = open(image_path, 'rb').read()
    elif isinstance(image_path,Image.Image):
        image_path = image_path.convert("RGB")
        buffered = BytesIO()
        image_path.save(buffered, format="png")
        image = buffered.getvalue()
        
    return base64.b64encode(image).decode('utf-8')


def get_content_between_a_b(start_tag, end_tag, text):
    extracted_text = ""
    start_index = text.find(start_tag)
    while start_index != -1:
        end_index = text.find(end_tag, start_index + len(start_tag))
        if end_index != -1:
            extracted_text += text[start_index + len(start_tag) : end_index] + " "
            start_index = text.find(start_tag, end_index + len(end_tag))
        else:
            break

    return extracted_text.strip()

def before_retry_fn(retry_state):
    if retry_state.attempt_number > 1:
        print(f"Retrying API call. Attempt #{retry_state.attempt_number}, f{retry_state}")


class openai_llm:
    def __init__(self,model = "gpt4o (05-13)") -> None:
        self.model = model
        deployment = ""
        endpoint = ""
        api_key = ""
        api_version = ""


        self.client = AzureOpenAI(
            azure_deployment = deployment,
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version= api_version
            )
        self.async_client = AsyncAzureOpenAI(
            azure_deployment = deployment,
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version= api_version
            )
    
    
    @retry(wait=wait_fixed(10), stop=stop_after_attempt(10), before=before_retry_fn)
    def response(self,messages,**kwargs):
        response = self.client.chat.completions.create(
            model=kwargs.get("model", self.model),
            messages=messages,
            n = kwargs.get("n", 1),
            temperature= kwargs.get("temperature", 0.0),
            max_tokens=kwargs.get("max_tokens", 4000),
            timeout=kwargs.get("timeout", 180)
        )
        return response.choices[0].message.content
    
    
    @retry(wait=wait_fixed(10), stop=stop_after_attempt(10), before=before_retry_fn)
    async def response_async(self,messages,**kwargs):
        response = await self.async_client.chat.completions.create(
            model=kwargs.get("model", self.model),
            messages=messages,
            n = kwargs.get("n", 1),
            temperature= kwargs.get("temperature", 0.0),
            max_tokens=kwargs.get("max_tokens", 4000),
            timeout=kwargs.get("timeout", 180)
        )
        return response.choices[0].message.content
    
    async def deal_tasks(self,tasks, max_concurrent_tasks=20):
        semaphore = asyncio.Semaphore(max_concurrent_tasks)
        results = []

        async def sem_task(task):
            async with semaphore:
                return await task 

        sem_tasks = [sem_task(task) for task in tasks]

        for coro in tqdm_asyncio.as_completed(sem_tasks, total=len(sem_tasks), desc="Processing tasks"):
            result = await coro
            results.append(result)

        return results
    

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

        content = []
        for frame,frame_time in zip(video,frame_times):
            content.append({"type":"text","text":f"{frame_time}s:"})
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(frame)}", "detail": "low"}})
        content.append({"type":"text","text":prompt})
        messages = [{"role":"user","content":content}]

        return messages


    def generate_output(self,messages):
        messages = self.process_messages(messages)
        return self.response(messages)
    
    async def generate_output_async(self,messages,idx):
        response = await self.response_async(messages)
        return response,idx
    
    def generate_outputs(self,messages_list):
        tasks = []
        for idx,messages in tqdm(enumerate(messages_list),desc="Processing messages",total=len(messages_list)):
            messages = self.process_messages(messages)
            tasks.append(self.generate_output_async(messages,idx))
        results = asyncio.run(self.deal_tasks(tasks))
        results = sorted(results,key = lambda x:x[1])
        results = [result[0] for result in results]
        return results

