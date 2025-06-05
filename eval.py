import torch
import os
import random
import logging
import numpy as np
from tqdm import tqdm
import json
from argparse import ArgumentParser
from LLMs import init_llm
from utils import eval_benchmark
import gc




def set_seed(seed_value):
    """
    Set the seed for PyTorch (both CPU and CUDA), Python, and NumPy for reproducible results.

    :param seed_value: An integer value to be used as the seed.
    """
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # For multi-GPU setups
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    parser = ArgumentParser()
    parser.add_argument('--eval_datasets', type=str, default='8_frames',
                    help='name of eval dataset')
    parser.add_argument('--datasets_path', type=str, default="datas",
                    help='path of eval dataset')
    parser.add_argument('--output_path', type=str, default='eval_results/Qwen2-VL-7B-Instruct',
                        help='name of saved json')
    parser.add_argument('--model_name', type=str, default='Qwen2-VL-7B-Instruct',
                        help='name of model')
    parser.add_argument('--model_path', type=str, default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_chunks', type=str, default="1")
    parser.add_argument('--chunk_idx', type=str, default="0")
    parser.add_argument('--cuda_visible_devices', type=str, default=None)
    parser.add_argument('--ann_file', type=str, default=None)


    args = parser.parse_args()
    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
        os.environ["tensor_parallel_size"] = str(len(args.cuda_visible_devices.split(",")))

    os.environ["num_chunks"] = args.num_chunks
    os.environ["chunk_idx"] = args.chunk_idx
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    os.makedirs(args.output_path, exist_ok=True)

    print('initializing LLM...')
    model = init_llm(args.model_name,args.model_path)
    eval_dataset = args.eval_datasets
    set_seed(args.seed)
    print(f'evaluating on {eval_dataset}...')
    eval_output_path = os.path.join(args.output_path,eval_dataset)
    eval_dataset_path = args.datasets_path
    print(eval_dataset_path)
    os.makedirs(eval_output_path, exist_ok=True)
    if not os.path.exists(eval_dataset_path):
        raise f"{eval_dataset_path} not exits"

    eval_benchmark(model,eval_dataset_path,eval_output_path,eval_dataset)
    print(f'final results on {eval_dataset}')
    gc.collect()

if __name__ == '__main__':
    main()
