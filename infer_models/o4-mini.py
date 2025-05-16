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
import copy
import time
import base64


DIRECT_PROMPT = "Directly answer the question with one option letter without explanation."
DIRECT_MM_PROMPT = "Solve the multiple-choice question in image. Directly answer the question with one option letter without explanation."
CoT_PROMPT = "Solve the multiple-choice question and then answer with one option letter. The last line of your response should be of the following format: 'Answer: LETTER' where LETTER is one of options. Think step by step before answering."
CoT_MM_PROMPT = "Solve the multiple-choice question in image and then answer with one option letter. The last line of your response should be of the following format: 'Answer: LETTER' where LETTER is one of options. Think step by step before answering."


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


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
    parser.add_argument("--base_url", type=str, default="your_base_url")
    parser.add_argument("--image_folder", type=str, default="./images")
    parser.add_argument("--json_file", type=str, default="LogicOCR.json")
    parser.add_argument("--output_filename", type=str, default="o4-mini")
    parser.add_argument("--output_folder", type=str, default="./res")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--verbose", action='store_true')
    args = parser.parse_args()
    return args


def get_response(data, args):
    retry_count = 2
    retry_interval = 1
    responsed_data = copy.deepcopy(data)
    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    
    for _ in range(retry_count):
        try:
            local_path = os.path.join(args.image_folder, data["image"])
            base64_image = encode_image(local_path)
            response = client.chat.completions.create(
                model="o4-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": CoT_MM_PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                            },
                        ]
                     },
                ], 
                max_tokens=8192, 
                temperature=0
            )
            content = response.choices[0].message.content
            responsed_data["response"] = content
            completion_tokens = response.usage.completion_tokens
            responsed_data["output_tokens"] = completion_tokens
            
            score = 0
            content_choice = parse_response(content, data["choices"])
            if content_choice!=None and content_choice.lower() == data["solution"].lower():
                score = 1
            if args.verbose:
                print(f'{content}\nCompletion tokens: {completion_tokens}, Prediction: {content_choice}, GT: {data["solution"]}, Score: {score}')
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
    
    if os.path.exists(log_path):
        result_list.extend(existed_data_list)
    result_list.sort(key=lambda x: x["id"])
    
    scores = [item["score"] for item in result_list if "score" in item]
    print(f"{len(scores)} samples, average score: {sum(scores)/len(scores)}")
    
    save_json(result_list, log_path)
    print(f"save the predictions to {log_path}")