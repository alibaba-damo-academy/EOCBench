# Model Evaluation Documentation

## Step 1: Data Preparation

1. **Download the [data](https://huggingface.co/datasets/CircleRadon/EOC-Bench)** to your local directory and set the `OUTPUT_PATH` environment variable to point to that location.

2. **Specify the evaluation datasets** by setting the `EVAL_DATASETS`. We currently support formats such as `"1_frames, 8_frames, 16_frames, 32_frames, 1_fps"`.

## Step 2: Model Configuration


1. **Configure the model settings** by specifying the `MODEL_NAME` and `MODEL_PATH`. The available models include:
   - `"gpt-4o"`
   - `"gemini"`
   - `"qwen2.5-vl"`  
   - `"internvl2_5"`  
   - `"llava_video"`  
   - `"llava_onevision"`  
   - `"video_llava"`  
   - `"longva"`  
   - `"videollama2"`  
   - `"videollama3"`  
   - `"nvila"`
   - `"videorefer"`
   - `"vip-llava"`
   - `"osprey"`
   - `"sphinx-v"`


## Step 3: Execute Evaluation

Run `bash eval.sh`.

There is an example of `eval.sh`:
```bash
EVAL_DATASETS="1_fps"

DATASETS_PATH="EOC-Bench/videos/videos"
MODEL_NAME="qwen2.5-vl"
OUTPUT_PATH="./eval_results"
MODEL_PATH="Qwen/Qwen2.5-VL-7B-Instruct"

SEED=42

python eval.py \
    --eval_datasets "$EVAL_DATASETS" \
    --datasets_path "$DATASETS_PATH" \
    --output_path "$OUTPUT_PATH" \
    --model_name "$MODEL_NAME" \
    --model_path "$MODEL_PATH" \
    --output_path "$OUTPUT_PATH" \
    --seed $SEED &
done

```

To expedite the evaluation, `multiple GPUs`` can be utilized for parallel processing. Below is an example configuration:

```bash
gpu_list="0,1,2,3,4,5,6,7"

IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

EVAL_DATASETS_LIST="1_fps"

DATASETS_PATH="EOC-Bench/videos/videos"
MODEL_NAME="qwen2.5-vl"
OUTPUT_PATH="./eval_relults"
MODEL_PATH="Qwen/Qwen2.5-VL-7B-Instruct"

SEED=42

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python eval.py \
    --eval_datasets "$EVAL_DATASETS" \
    --datasets_path "$DATASETS_PATH" \
    --output_path "$OUTPUT_PATH" \
    --model_name "$MODEL_NAME" \
    --model_path "$MODEL_PATH" \
    --output_path "$OUTPUT_PATH" \
    --num_chunks $CHUNKS \
    --chunk_idx $IDX \
    --seed $SEED &
done
wait
```


## 🔧 Support New Models


To support your own models, you can define a new class in the `models`` folder. Below is a template to guide you:

```python
class YourOwnModel
    def __init__(self, model_path):
        # load your model and processor here
    def generate_outputs(self, messages_list):
        # input: a list of messages
        # output the model answers in a list
```

Format of messages_list:
```md
[
    {
        'idx': index,
        'video': video_path,
        'prompt': question,
        'answer': ['A', 'B'],
        'video_time': video time,
        'box' : [[xx,xx,xx,xx],],
        'frame_type': '1fps/32frames/',
        'fps': 60,
        'frame_number': raw frame number
    }
]
```

###  NOTE:

We offer a `load_video` function for sampling video frames. For general models, a visual prompt is added to the last frame of the video. Ensure the last frame is sampled by setting `add_last_frame=True`.

For object-level models that can handle visual prompts through an object encoder, avoid sampling the last frame by setting `add_last_frame=False`.

🔥 Feel free to submit a pull request if you develop new models.