from google import genai
from tqdm import tqdm
from PIL import Image
import base64
from tenacity import (
    retry,
    stop_after_attempt,
    wait_fixed,
)
from PIL import Image
from io import BytesIO

from .utils import load_video


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



class Gemini:
    def __init__(self,model = "gemini-2.0-flash") -> None:
        self.model = model
        self.api_idx = 0
        self.api_pools = [
            "your api key"
        ]

        self.client = genai.Client(api_key=self.api_pools[self.api_idx])
    
    @retry(wait=wait_fixed(10), stop=stop_after_attempt(3), before=before_retry_fn)
    def response(self,messages,**kwargs):
        response = self.client.models.generate_content(
            model=kwargs.get("model", self.model),
            contents=messages,
        )
        return response.text
    
    


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

        contents = []
        for image,frame_time in zip(video,frame_times):
            if image.size[0] * image.size[1] >= 1024 * 512:
                image = image.resize((1024,512))
            contents.append(f"{frame_time}s:")
            contents.append(image)
        contents.append(prompt)
        return contents

    def generate_output(self,messages,**kwargs):
        messages = self.process_messages(messages)
        response = None
        cnt = 0
        while response is None:
            try:
                response = self.response(messages)
            except Exception as e:
                self.api_idx = (self.api_idx + 1) % len(self.api_pools)
                self.client = genai.Client(api_key=self.api_pools[self.api_idx])
                model = kwargs.get("model", self.model)
                print(f"get {model} response failed: {e}")
                response = None
            cnt += 1
            assert cnt < 21, "error while retry more than 20 times"
        return response
    
    def generate_outputs(self,messages_list,**kwargs):
        results = []
        for messages in tqdm(messages_list):
            result = self.generate_output(messages,**kwargs)
            results.append(result)
        return results

