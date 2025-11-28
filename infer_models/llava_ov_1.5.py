import sys
import json
from argparse import ArgumentParser
from tqdm import tqdm
import re
import torch
import multiprocessing
from multiprocessing import Pool, Queue, Manager
import os
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from qwen_vl_utils import process_vision_info
import copy
import random
random.seed(1234)


DIRECT_MM_PROMPT = "Solve the multiple-choice question in image. Directly answer the question with one option letter without explanation."
CoT_MM_PROMPT = "Solve the multiple-choice question in image and then answer with one option letter. The last line of your response should be of the following format: 'Answer: LETTER' where LETTER is one of options. Think step by step before answering."


def split_list(lst, n):
    length = len(lst)
    avg = length // n
    result = []
    for i in range(n - 1):
        result.append(lst[i*avg:(i+1)*avg])
    result.append(lst[(n-1)*avg:])
    return result 


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
    
    if "The answer is " in response:
        return response.split("The answer is ")[-1].replace('.', '')
    if "The correct answer is " in response:
        return response.split("The correct answer is ")[-1].replace('.', '')
    
    return None


def _get_args():
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, default="lmms-lab/LLaVA-OneVision-1.5-8B-Instruct")
    parser.add_argument("--image_folder", type=str, default="./images")
    parser.add_argument("--json_file", type=str, default="LogicOCR.json")
    parser.add_argument("--output_folder", type=str, default="./res")
    parser.add_argument("--lmm_input_modal", type=str, default="image-text")  # "text"
    parser.add_argument("--answer_directly", action='store_true')
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--verbose", action='store_true')
    args = parser.parse_args()
    return args


def infer_worker(args, data, eval_id, output_queue):
    device =  f"cuda:{eval_id}"
    image_root = args.image_folder
    input_modal = args.lmm_input_modal
    model_path = args.model_path
    verbose = args.verbose
    
    model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch.bfloat16,
            device_map=device,
            attn_implementation="flash_attention_2",
            trust_remote_code=True
        ).eval()
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    
    generate_kwargs = dict(
            max_new_tokens=2048,
            do_sample=False,
            use_cache=True,
        )

    for item in tqdm(data):
        if input_modal == "image-text":
            image_path = os.path.join(image_root, item["image"])
            # image = Image.open(image_path).convert("RGB")
            
            if args.answer_directly:
                prompt = DIRECT_MM_PROMPT
            else:
                prompt = CoT_MM_PROMPT
            
            messages = [
                {
                    "role": "user", 
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(model.device)
            generated_ids = model.generate(**inputs, **generate_kwargs)
            generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
            response = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
        else:
            print("Unknown input modal.")
            raise ValueError
        
        item["response"] = response
        
        score = 0
        response_choice = parse_response(response, item["choices"])
        if response_choice!=None and response_choice.lower() == item["solution"].lower():
            score = 1
        if verbose:
            print(f'{response}\nPrediction: {response_choice}, GT: {item["solution"]}, Score: {score}')
        item["score"] = score
        
    output_queue.put({eval_id: data})
    print(f"Process {eval_id} has completed.")
        

if __name__ == "__main__":
    args = _get_args()
    model_name = args.model_path.split('/')[-1]
    print(f"Evaluating: {model_name}")
    print(f"Input modal: {args.lmm_input_modal}")
    if args.answer_directly:
        print("Answer type: Answer directly")
        answer_mode = 'direct'
    else:
        print("Answer type: CoT")
        answer_mode = 'cot'
    
    multiprocessing.set_start_method('spawn')
    
    os.makedirs(args.output_folder, exist_ok=True)
    with open(args.json_file, 'r') as f:
        data = json.load(f)
    data_list = split_list(data, args.num_workers)
    
    output_queue = Manager().Queue()
    pool = Pool(processes=args.num_workers)
    for i in range(len(data_list)):
        pool.apply_async(
            infer_worker, 
            args=(args, data_list[i], i, output_queue)
            )
    pool.close()
    pool.join()
    # infer_worker(args, data_list[0], 0, output_queue)
    
    results = {}
    while not output_queue.empty():
        result = output_queue.get()
        results.update(result)
    data = []
    for i in range(len(data_list)):
        data.extend(results[i])
    
    
    scores = [dd["score"] for dd in data]
    print(f"Average score: {sum(scores)/len(scores)}")
    
    log_path = os.path.join(args.output_folder, model_name + f'_{args.lmm_input_modal}_{answer_mode}.json')
    save_json(data, log_path)
    print(f"save the predictions to {log_path}")