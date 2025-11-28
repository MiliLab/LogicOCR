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
import random
random.seed(1234)
import copy
import time
import base64


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def save_json(json_list, save_path):
    with open(save_path, 'w') as file:
        json.dump(json_list, file, indent=4)


def _get_args():
    parser = ArgumentParser()
    parser.add_argument("--api_key", type=str, default="your_api_key")
    parser.add_argument("--base_url", type=str, default="your_base_url")
    parser.add_argument("--image_folder", type=str, default="./LogicOCR_real")
    parser.add_argument("--json_file", type=str, default="LogicOCR_real.json")
    parser.add_argument("--output_filename", type=str, default="claude-3-7-sonnet-20250219")
    parser.add_argument("--output_folder", type=str, default="./res_real")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("--answer_directly", action='store_true')
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
            if args.answer_directly:
                prompt = data["question"] + " Directly answer the question using a single word or phrase without explanation."
            else:
                prompt = data["question"] + " The last line of your response should be of the following format: 'Answer: YOUR_ANSWER' where YOUR_ANSWER is the final answer. Think step by step before answering."
            response = client.chat.completions.create(
                model="claude-3-7-sonnet-20250219",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                            },
                        ]
                     },
                ], 
                max_tokens=8192, 
                temperature=0,
            )
            content = response.choices[0].message.content
            responsed_data["response"] = content
            # completion_tokens = response.usage.completion_tokens
            # responsed_data["output_tokens"] = completion_tokens
            
            if args.verbose:
                print(f'{content}\nGT: {data["solution"]}')
                
            print(f'id: {data["id"]} finished!\n\n')
            
            return responsed_data
        
        except Exception as e:
            print("ID: ", data["id"], " Error: ", e)
            print("Request again...")
            time.sleep(retry_interval)
        
    return data


if __name__ == "__main__":
    args = _get_args()
    print("Evaluating: ", args.output_filename)
    
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
    
    save_json(result_list, log_path)
    print(f"save the predictions to {log_path}")