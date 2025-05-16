from openai import OpenAI
import base64
import sys
import json
from tqdm import tqdm
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import collections
import time
from argparse import ArgumentParser
import re
import os
from dashscope import MultiModalConversation
import copy
import time


DIRECT_PROMPT = "Directly answer the question with one option letter without explanation."
DIRECT_MM_PROMPT = "Solve the multiple-choice question in image. Directly answer the question with one option letter without explanation."
CoT_PROMPT = "Solve the multiple-choice question and then answer with one option letter. The last line of your response should be of the following format: 'Answer: LETTER' where LETTER is one of options. Think step by step before answering."
CoT_MM_PROMPT = "Solve the multiple-choice question in image and then answer with one option letter. The last line of your response should be of the following format: 'Answer: LETTER' where LETTER is one of options. Think step by step before answering."


def save_json(json_list, save_path):
    with open(save_path, 'w') as file:
        json.dump(json_list, file, indent=4)


def parse_response(response, ans: str):
    """
    Return the last letter appearing after 'ANSWER:' in the input text.
    If there's no match, return None.
    """
    all_choices = ['A', 'B', 'C', 'D']
    response = response.replace('### Final Answer:', '')  # the 'Answer:' part affects answer extraction 
    
    pattern = r'Answer:\s*([A-Za-z])' # Answer: A
    matches = re.findall(pattern, response)
    if matches: 
        for match in matches[::-1]:
            if match in all_choices or match.upper() in all_choices: return match
    pattern = r'(?<!Final )Answer:\s*([A-Za-z])' # Answer: A
    matches = re.findall(pattern, response)
    if matches: 
        for match in matches[::-1]:
            if match in all_choices or match.upper() in all_choices: return match
    pattern = r'Answer:\s*\*\*([A-Za-z])\*\*' # Answer: **A**
    matches = re.findall(pattern, response)
    if matches: 
        for match in matches[::-1]:
            if match in all_choices or match.upper() in all_choices: return match
    pattern = r'Answer:\s*\$([A-Za-z])\$' # Answer: $A$
    matches = re.findall(pattern, response)
    if matches: 
        for match in matches[::-1]:
            if match in all_choices or match.upper() in all_choices: return match
    pattern = r'Answer:\s*\$([A-Za-z])' # Answer: $A
    matches = re.findall(pattern, response)
    if matches: 
        for match in matches[::-1]:
            if match in all_choices or match.upper() in all_choices: return match
    pattern = r'\\boxed\{([A-Za-z])\}'
    matches = re.search(pattern, response, re.DOTALL)
    if matches:
        return matches.group(1).strip()
    
            
    pattern = r'\s*\(([A-Za-z])\)'  # e.g., (A) (B) (C) (D)
    matches = re.findall(pattern, response)
    if matches: 
        for match in matches[::-1]:
            if match in all_choices or match.upper() in all_choices: return match
    response = ' ' + response.strip()
    pattern = r'\s*([A-Za-z])\)'   # e.g., A) B) C) D)
    matches = re.findall(pattern, response)
    if matches: 
        for match in matches[::-1]:
            if match in all_choices or match.upper() in all_choices: return match
    pattern = r'\s*\{([A-Za-z])\}' # e.g., {A} {B} {C} {D}
    matches = re.findall(pattern, response)
    if matches: 
        for match in matches[::-1]:
            if match in all_choices or match.upper() in all_choices: return match
    pattern = r'\s*\$([A-Za-z])\$' # e.g., $A$, $B$, $C$, $D$
    matches = re.findall(pattern, response)
    if matches: 
        for match in matches[::-1]:
            if match in all_choices or match.upper() in all_choices: return match
    pattern = r" ([A-Da-d])\." # e.g., A. B. C. D.
    matches = re.findall(pattern, response)
    if matches: 
        for match in matches[::-1]:
            if match in all_choices or match.upper() in all_choices: 
                return match
    pattern = r" ([A-Da-d])" # e.g., A B C D
    matches = re.findall(pattern, response)
    if matches and len(response) <= 5: 
        for match in matches[::-1]:
            if match in all_choices or match.upper() in all_choices: 
                return match
    if ans != None:
        ans = ans.split('\n')
        for choice in ans:
            if f'answer: {choice.lower()}' in response.lower(): return choice[0]
            if f'answer:{choice.lower()}' in response.lower(): return choice[0]
    return None


def _get_args():
    parser = ArgumentParser()
    parser.add_argument("--api_key", type=str, default="your_api_key")
    parser.add_argument("--image_folder", type=str, default="./images")
    parser.add_argument("--json_file", type=str, default="LogicOCR.json")
    parser.add_argument("--output_filename", type=str, default="qvq_72b_preview")
    parser.add_argument("--output_folder", type=str, default="./res")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--verbose", action='store_true')
    args = parser.parse_args()
    return args


def get_response(data, args):
    retry_count = 2
    retry_interval = 1
    responsed_data = copy.deepcopy(data)
    gen_kwargs = dict(
            max_length=8192,
            temperature=0,
            seed=1234,
        )
    
    for _ in range(retry_count):
        try:
            local_path = os.path.join(args.image_folder, data["image"])
            image_path = f"file://{local_path}"
            prompt = CoT_MM_PROMPT
            messages = [
                {
                    "role": "system",
                    "content": [
                        {"text": "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step."}
                        ]
                },
                {
                    'role':'user',
                    'content': [
                        {'image': image_path},
                        {'text': prompt}
                        ]
                }
            ]
            response = MultiModalConversation.call(
                api_key=args.api_key,
                model='qvq-72b-preview',
                messages=messages,
                vl_high_resolution_images=True,
                **gen_kwargs
            )
            content = response["output"]["choices"][0]["message"].content[0]["text"]
            responsed_data["response"] = content
            responsed_data["output_tokens"] = response.usage.output_tokens
            
            score = 0
            content_choice = parse_response(content, data["choices"])
            if content_choice!=None and content_choice.lower() == data["solution"].lower():
                score = 1
            if args.verbose:
                print(f'{content}\nPrediction: {content_choice}, GT: {data["solution"]}, Score: {score}')
            responsed_data["score"] = score
            print(f'id: {data["id"]} finished!\n\n')
            return responsed_data
        
        except Exception as e:
            print("ID: ", data["id"], " Error: ", e)
            print("Request again...")
            time.sleep(retry_interval)
        
    return data


if __name__ == "__main__":
    args = _get_args()
      
    os.makedirs(args.output_folder, exist_ok=True)
    read_path = args.json_file  
    with open(read_path, 'r') as f:
        data_list = json.load(f)
    
    log_path = os.path.join(args.output_folder, args.output_filename+'.json')
    if os.path.exists(log_path):
        with open(log_path, 'r') as f_exist:
            existed_data_list = json.load(f_exist)
        existed_data_list = [exist_item for exist_item in existed_data_list if "response" in exist_item]
        print(f'Have achieved {len(existed_data_list)} answered samples.')
        existed_id_list = [exist_item["id"] for exist_item in existed_data_list]
        data_list = [item for item in data_list if item["id"] not in existed_id_list]
        if len(data_list) == 0:
            print('Finished')
            sys.exit()
        print(f'Only get responses for {len(data_list)} samples.')
    
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [executor.submit(get_response, data, args) for data in data_list]
        result_list = []
        for job in as_completed(futures):
            response_dict = job.result(timeout=None)
            result_list.append(response_dict)
            time.sleep(1)
    
    if os.path.exists(log_path):
        result_list.extend(existed_data_list)
    result_list.sort(key=lambda x: x["id"])
    
    save_json(result_list, log_path)
    print(f"save the predictions to {log_path}")
    
    scores = [item["score"] for item in result_list if "score" in item]
    print(f"{len(scores)} samples, average score: {sum(scores)/len(scores)}")