import torch
import os
import random
import difflib 
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import re
import math
import json
import csv
import gc
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from datasets import load_dataset

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def save_json(filename, ds):
    with open(filename, 'w') as f:
        json.dump(ds, f, indent=4)

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


def extract(text, type, hard = True):
    if text:
        target_str = get_content_between_a_b(f"<{type}>", f"</{type}>", text)
        if target_str:
            return target_str
        elif hard:
            return text
        else:
            return ""
    else:
        return ""

def find_first_number(s):
    pattern = r'\d+\.?\d*'
    match = re.search(pattern, s)
    if match:
        return match.group()
    else:
        return None
    
def extract_time(text):
    def replace_object_tags(text):
        result = re.sub(r'<object \d+>', '', text)
        return result
    text = replace_object_tags(text)
    number = find_first_number(text)
    if number is None:
        return 0
    return number

def calculate_time_awareness_score(gt, pred, thresholds=None):
    gt, pred = float(gt), float(pred)
    errors = [0.01, 0.1, 0.2, 0.3]
    accurate_counts = [0] * len(errors)
    
    error = abs(gt - pred)
    
    for i, threshold in enumerate(errors):
        if error <= threshold * gt:
            accurate_counts[i] += 1

    score = sum(accurate_counts) / len(errors)
    return score


def run_model(samples, model,save_path = None):
    out_samples = []
    f = open(save_path, "w")
    with torch.no_grad():
        messages_list = []
        current_messages = []
        current_samples = []
        for sample in samples:
            messages = sample["messages"]
            current_messages.append(messages)
            current_samples.append(sample)
            if len(current_messages) >= 1:
                messages_list.append([current_messages,current_samples])
                current_messages = []
                current_samples = []
        if current_messages:
            messages_list.append([current_messages,current_samples])
        
        for current_messages,current_samples in tqdm(messages_list):
            success = True
            try:
                outputs = model.generate_outputs(current_messages)
            # print(outputs)
            except Exception as e:
                print(e)
                outputs = ["A"] * len(current_samples)
                success = False
            for sample,response in zip(current_samples,outputs):
                del sample["messages"]
                sample["response"] = response
                sample["success"] = success
                out_samples.append(sample) 
                # with open(save_path, 'w') as f:
                #     f.write(json.dumps(out_samples, indent=4))
                f.write(json.dumps(sample)+"\n")
            gc.collect()  
    f.close()
    return out_samples

def construct_messages(sample, frame_type, visual_prompt='box'):
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink']
    question = sample["question"]
    answer = sample["answer"]
    video = sample["video"]
    idx = sample["idx"]
    fps = sample["fps"]
    frame_number = sample["frame_number"]

    dataset_path = sample["dataset_path"]
    idx = sample["idx"]

    video = os.path.join(dataset_path,video)

    if "choices" in sample:
        raw_choices = sample["choices"]
        choices = [f"{option.upper()}. {choice}" for option,choice in raw_choices.items()]
        options = "\n".join(choices)
    else:
        raw_choices = None

    pattern = r'<object \d>'

    matches = re.findall(pattern, question)
    
    if sample['video_type'] == 'Absolute Time Perception':
        prompt_str = f"I have overlaid the {visual_prompt} on the last frame of the video, "
        for i in range(len(matches)):
            prompt_str+=f"<object {i}>:{colors[i]}; "
        prompt_str = prompt_str[:-2]+'. '
        prompt = f"""
Question: {prompt_str}{question} Please output the answer directly in seconds.
"""

    elif len(matches)>0:
        prompt_str = f"I have overlaid the {visual_prompt} on the last frame of the video, "
        for i in range(len(matches)):
            prompt_str+=f"<object {i}>:{colors[i]}; "
        prompt_str = prompt_str[:-2]+'. '
        # print(prompt_str+question)

        prompt = f"""
Question: {prompt_str}{question}
Options: 
{options}
"""
        if len(answer) == 1:
            # prompt += "Answer directly using the letters of the options given and wrap your response in <choice></choice>. For example, if the answer is A, then output <choice>A</choice>"
            prompt += "Answer directly using the letters of the options given and wrap your response."
        else:
            prompt += "Answer directly using the letters of the options given. There are multiple answers, so wrap your response in <choice></choice>. For example, if the answer is A and B, then output <choice>A, B</choice>; if the answer is A, B and C, then output <choice>A, B, C</choice>"

    else:
        prompt = f"""
Question: {question}
Options: 
{options}
"""

        if len(answer) == 1:
            prompt += "Answer directly using the letters of the options given and wrap your response in <choice></choice>. For example, if the answer is A, then output <choice>A</choice>"
        else:
            prompt += "Answer directly using the letters of the options given. There are multiple answers, so wrap your response in <choice></choice>. For example, if the answer is A and B, then output <choice>A, B</choice>; if the answer is A, B and C, then output <choice>A, B, C</choice>"

    messages = {
        "prompt":prompt,
        "video":video,
        "answer":answer,
        "choices":raw_choices,
        "idx":idx, 
        "video_time":sample["video_time"], 
        "video_type": sample["video_type"], 
        "box": sample["box"],
        "frame_type": frame_type, 
        "fps": fps,
        'frame_number': frame_number
    }
    sample["messages"] = messages
    return sample


def str_similarity(str1, str2):
    seq = difflib.SequenceMatcher(None, str1, str2)
    return seq.ratio()

def find_most_similar_index(str_list, target_str):
    """
    Given a list of strings and a target string, returns the index of the most similar string in the list.
    """
    most_similar_index = 0
    highest_similarity = 0

    for i, s in enumerate(str_list):
        similarity = str_similarity(s, target_str)
        
        if similarity > highest_similarity:
            most_similar_index = i
            highest_similarity = similarity
    
    return most_similar_index

def cal_metrics(out_samples):
    key_mapping = {
        "Object State Retrospection": "Past",
        "Location Retrospection": "Past",
        "Object Relationship Evolution": "Past",
        "Absolute Time Perception": "Past",
        "Immediate State Recognition": "Present",
        "Object Relationship": "Present",
        "Purpose and Function Inference": "Present",
        "Anomaly Perception": "Present",
        "Trajectory and Motion Prediction": "Future",
        "State Change Prediction": "Future",
        "Dynamic Relationship Prediction": "Future"
    }
    total_cnt = defaultdict(float)
    total_right = defaultdict(float)
    total_single = defaultdict(float)
    total_multi = defaultdict(float)
    single_right = defaultdict(float)
    multi_right = defaultdict(float)
    adjust_right = defaultdict(float)
    total_failed = defaultdict(float)
    for i,out_sample in enumerate(out_samples):
        name = out_sample["video_type"]
        answer = out_sample["answer"]
        response = out_sample["response"]
        success = out_sample["success"]
        answer = [item.lower() for item in answer]
        if isinstance(response, list):
            response = "a"
        response = response.lower()
        orig_response = response
        if "<s>" in response:
            response = extract(response,"s")
        # response = response[0]
        alphas = ["a","b","c","d","e"]
        try:
            if 'choice_type' in out_sample and out_sample['choice_type']=="open-ended":
                response = extract_time(response)
            else:
                response = extract(response,"choice")
                if "." in response:
                    response = response.split(".")[0]
                response = [item.strip() for item in response.split(",")]
                for r in response:
                    if r not in alphas:
                        options = [f"{key}. {value}" for key,value in out_sample["choices"].items()]
                        response = find_most_similar_index(options, orig_response)
                        response = [alphas[response]]
                        break
        except:
            response = ["a"]

        if not success:
            total_failed[name] += 1
            total_failed[key_mapping[name]] += 1
            total_failed["total"] += 1

        if len(answer) == 1:
            total_single["total"] += 1
            total_single[name] += 1
            total_single[key_mapping[name]] += 1
        else:
            total_multi["total"] += 1
            total_multi[name] += 1
            total_multi[key_mapping[name]] += 1
        
        
        if 'choice_type' in out_sample and out_sample['choice_type']=='open-ended':
            time_answer = find_first_number(answer[0])
            time_score = calculate_time_awareness_score(time_answer, response)
            total_right["total"] += time_score
            total_right[name] += time_score
            total_right[key_mapping[name]] += time_score
        else:
            if sorted(response) == sorted(answer):
                total_right["total"] += 1
                total_right[name] += 1
                total_right[key_mapping[name]] += 1
                if len(answer) == 1:
                    single_right["total"] += 1
                    single_right[name] += 1
                    single_right[key_mapping[name]] += 1
                    out_samples[i]["correct"] = True
                else:
                    multi_right["total"] += 1
                    multi_right[name] += 1
                    multi_right[key_mapping[name]] += 1
                    out_samples[i]["correct"] = True
            else:
                out_samples[i]["correct"] = False
        
            response = [option.lower() for option in response]
            answer = [option.lower() for option in answer]
            partial_right = 0
            for option in response:
                if option in answer:
                    partial_right += 1/(len(answer))
                else:
                    partial_right = 0
                    break
            adjust_right[name] += partial_right
            adjust_right[key_mapping[name]] += partial_right
            adjust_right["total"] += partial_right
        total_cnt["total"] += 1
        total_cnt[name] += 1
        total_cnt[key_mapping[name]] += 1
    final_scores = {}
    # f.close()
    for name in total_cnt.keys():
        final_scores[name] = {
            "total_acc": total_right[name]/total_cnt[name],
            "single_acc": single_right[name]/(total_single[name]+1e-6),
            "multi_acc": multi_right[name]/(total_multi[name]+1e-6),
            "adjust_acc": adjust_right[name]/total_cnt[name],
            "total cnt": total_cnt[name],
            "right cnt": total_right[name],
            "single cnt": total_single[name],
            "right single cnt": single_right[name],
            "multi right cnt": multi_right[name],
            "multi cnt": total_multi[name],
            "failed cnt": total_failed[name]
        }
    return final_scores,out_samples

def cal_metrics_by_name(out_samples):
    key_mapping = {
        "Object State Retrospection": "Past",
        "Location Retrospection": "Past",
        "Object Relationship Evolution": "Past",
        "Absolute Time Perception": "Past",
        "Immediate State Recognition": "Present",
        "Object Relationship": "Present",
        "Purpose and Function Inference": "Present",
        "Anomaly Perception": "Present",
        "Trajectory and Motion Prediction": "Future",
        "State Change Prediction": "Future",
        "Dynamic Relationship Prediction": "Future"
    }
    nums = defaultdict(lambda: defaultdict(float))
    score_sums = defaultdict(lambda: defaultdict(float))
   
    for out_sample in out_samples:
        name = out_sample["video_type"]
        answer = out_sample["answer"]
        time_answer = answer[0]
        response = out_sample["response"]
        success = out_sample["success"]
        answer = [item.lower() for item in answer]
        if isinstance(response, list):
            response = "a"
        response = response.lower()
        orig_response = response
        
        if "choice_type" in out_sample:
            choice_type = out_sample["choice_type"]
        else:
            if len(answer)>1:
                choice_type = 'multi-choice'
            elif len(answer)==2:
                choice_type = 'true/false'
            else:
                choice_type = 'single-choice'
        # response = response[0]
        if "<s>" in response:
            response = extract(response,"s")
        answer = [item.lower() for item in answer]
        alphas = ["a","b","c","d","e"]
        try:
            if 'choice_type' in out_sample and out_sample['choice_type']=="open-ended":
                response = extract_time(response)
            else:
                response = extract(response,"choice")
                if "." in response:
                    response = response.split(".")[0]
                response = [item.strip() for item in response.split(",")]
                for r in response:
                    if r not in alphas:
                        options = [f"{key}. {value}" for key,value in out_samples["choices"].items()]
                        response = find_most_similar_index(options,orig_response)
                        response = [alphas[response]]
                        break
        except:
            response = ["a"]
        
        nums[key_mapping[name]][choice_type]+=1
        nums['mean'][choice_type]+=1
    
        if 'choice_type' in out_sample and out_sample['choice_type']=='open-ended':
            time_answer = find_first_number(time_answer)
            time_score = calculate_time_awareness_score(time_answer, response)
            score_sums["mean"][choice_type] += time_score
            score_sums[key_mapping[name]][choice_type] += time_score
        else:
            if sorted(response) == sorted(answer):
                score_sums["mean"][choice_type] += 1
                score_sums[key_mapping[name]][choice_type] += 1
       
    final_scores = {}
    # f.close()
    for name in score_sums.keys():
        final_scores[name] = {
            "single_acc": score_sums[name]['single-choice']/(nums[name]['single-choice']+1e-6),
            "multi_acc": score_sums[name]['multi-choice']/(nums[name]['multi-choice']+1e-6),
            "judge_acc": score_sums[name]['true/false']/(nums[name]['true/false']+1e-6),
        }
        if nums[name]['open-ended']>0:
            final_scores[name]["time_acc"] = score_sums[name]['open-ended']/(nums[name]['open-ended']+1e-6)
    return final_scores

def save_metrics2excel(data, output_path):
    key_mapping = {
        "total": "Overall",
        "Object State Retrospection": "OSR",
        "Location Retrospection": "LR",
        "Object Relationship Evolution": "ORE",
        "Absolute Time Perception": "ATP",
        "Past": "Past",
        "Immediate State Recognition": "ISR",
        "Object Relationship": "OR",
        "Purpose and Function Inference": "PFI",
        "Anomaly Perception": "AP",
        "Present": "Present",
        "Trajectory and Motion Prediction": "TMP",
        "State Change Prediction": "SCP",
        "Dynamic Relationship Prediction": "DRP",
        "Future": "Future"
    }

    # Create a dictionary for the Excel structure
    excel_data = {}

    # Populate the Excel data structure
    for key, acronym in key_mapping.items():
        if key in data:
            excel_data[acronym] = round(data[key]["total_acc"] * 100, 2)
        else:
            excel_data[acronym] = None  # or 0 if you prefer

    # Create a DataFrame
    df = pd.DataFrame([excel_data], index=['Total Acc (%)'])
    df.to_excel(os.path.join(output_path, 'matics.xlsx'))


def save_metrics2excel_1(data, output_path):
    flattened_data = {}
    cats = ['mean', 'Past', 'Present', 'Future']
    # for category, metrics in data.items():
    for cat in cats:
        metrics = data[cat]
        for metric, value in metrics.items():
            key = f"{cat}_{metric}"
            flattened_data[key] = round(value*100, 2)

    df = pd.DataFrame([flattened_data])
    df.to_excel(os.path.join(output_path, 'matics_cls.xlsx'))


def load_dataset(dataset_path, ann_file="meta_infos.json",is_one_frames = False):
    datasets = []

    json_path = os.path.join(dataset_path,ann_file)
    with open(json_path,"r") as f:
        datas = json.load(f)
    
    for idx,data in enumerate(datas):
        data['video'] = data['video_path']
        data["dataset_path"] = dataset_path

        datasets.append(data)
    chunks_num = int(os.environ.get("num_chunks",1))
    chunk_idx = int(os.environ.get("chunk_idx",0))
    print(f"chunks_num: {chunks_num}, chunk_idx: {chunk_idx}")
    datasets = get_chunk(datasets,chunks_num,chunk_idx)
    return datasets


def eval_benchmark(model,dataset_path,output_path,frame_type):
    dataset = load_dataset(dataset_path)
    samples = []
    for sample in dataset:
        sample = construct_messages(sample,frame_type)
        samples.append(sample)

    chunk_idx = int(os.environ.get("chunk_idx",0))
    num_chunks = int(os.environ.get("num_chunks",1))

    if num_chunks > 1:
        results_path = os.path.join(output_path,f"results_{chunk_idx}.json")
        out_samples = run_model(samples,model,results_path)
        # save_json(results_path,out_samples)

        total_results_path = os.listdir(output_path)
        total_results_path = [result for result in total_results_path if result.startswith("results_")]
        if len(total_results_path) == num_chunks:
            total_results = []
            for result in total_results_path:
                results_path = os.path.join(output_path,result)
                with open(results_path,"r") as f:
                    extend_datas = f.readlines()
                    extend_datas = [json.loads(data) for data in extend_datas]
                    total_results.extend(extend_datas)
            with open(os.path.join(output_path,"results.json"),"w") as f:
                json.dump(total_results,f,indent=4)
            metrics,out_samples = cal_metrics(total_results)
            metric_path = os.path.join(output_path,"metrics.json")
            save_json(metric_path,metrics)
            save_json(os.path.join(output_path,"results.json"),out_samples)
            save_metrics2excel(metrics, output_path)
            return metrics
        else:
            return None
    elif num_chunks == 1:
        results_path = os.path.join(output_path,"results.json")
        out_samples = run_model(samples,model,results_path)
        save_json(results_path,out_samples)
        metrics,out_samples = cal_metrics(out_samples)
        save_json(os.path.join(output_path,"results.json"),out_samples)
        metric_path = os.path.join(output_path,"metrics.json")
        save_json(metric_path,metrics)
        save_metrics2excel(metrics, output_path)
        print('evaluation finished...')
        return metrics
    else:
        raise ValueError("num_chunks must be greater than 0")

