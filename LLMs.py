
messages_example = [
    {"role": "system", "content": "You are a helpful assistant."},
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "cat.png",
            },
            {"type": "text", "text": "What is the text in the illustrate?"},
            {
                "type": "image",
                "image": "image(PIL.image)",
            },
        ],
    },
]

def init_llm(llm_name,model_path):
    llm = None
    llm_name = llm_name.lower()
    llm_names = ["qwen2.5-vl","internvl2_5","llava_video","llava_onevision","video_llava","longva","videollama2","videollama3","gpt","nvila"]
    if "qwen2.5-vl" in llm_name:
        from models.Qwen2_5_VL.Qwen2_5_VL import Qwen2_5_VL
        llm = Qwen2_5_VL(model_path)
    elif "internvl2_5" in llm_name:
        from models.InternVL.InternVL2_5 import InternVL
        llm = InternVL(model_path)
    elif "llava_video" in llm_name or "llava_onevision" in llm_name:
        from models.Llava_Video.Llava_Video import LlavaVideo
        llm = LlavaVideo(model_path)
    elif "video_llava" in llm_name:
        from models.Video_LLava.Video_LLava import VideoLLava
        llm = VideoLLava(model_path)
    elif "longva" in llm_name:
        from models.LongVA.LongVA import LongVA
        llm = LongVA(model_path)
    elif "videollama2" in llm_name:
        from models.VideoLlama2.Video_LLama import VideoLLama
        llm = VideoLLama(model_path)
    elif "videollama3" in llm_name:
        from models.VideoLlama3.Video_LLama import VideoLLama
        llm = VideoLLama(model_path)    
    elif "gpt" in llm_name:
        from models.GPT.GPT import openai_llm
        llm = openai_llm(model_path)
    elif "gemini" in llm_name:
        from models.Gemini.Gemini import Gemini
        llm = Gemini(model_path)
    elif "nvila" in llm_name:
        from models.Nvila.nvila import NVILA
        llm = NVILA(model_path)
    elif "videorefer" in llm_name:
        from models.VideoRefer.VideoRefer import VideoRefer
        llm = VideoRefer(model_path)
    elif "osprey" in llm_name:
        from models.Osprey.Osprey import Osprey
        llm = Osprey(model_path)
    elif "vipllava" in llm_name:
        from models.VIP_llava.VIP_llava import VIP_LLava
        llm = VIP_LLava(model_path)
    elif "sphinx_v" in llm_name:
        from models.SPHINX_V.SPHINX import SPHINX_V
        llm = SPHINX_V(model_path)
    else:
        raise f"{llm_name} not supported"
    return llm


