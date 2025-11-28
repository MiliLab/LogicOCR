import sys
import json
from argparse import ArgumentParser
from tqdm import tqdm
import re
from transformers import AutoModelForCausalLM
import torch
from PIL import Image
import multiprocessing
from multiprocessing import Pool, Queue, Manager
import os
import random
random.seed(1234)


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

            
def _get_args():
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, default="AIDC-AI/Ovis2.5-2B")
    parser.add_argument("--image_folder", type=str, default="./LogicOCR_real")
    parser.add_argument("--json_file", type=str, default="LogicOCR_real.json")
    parser.add_argument("--output_folder", type=str, default="./res_real")
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
            trust_remote_code=True,
            device_map=device
        ).eval()
    generate_kwargs = dict(
            max_new_tokens=2048,
            do_sample=False,
        )

    for item in tqdm(data):
        if input_modal == "image-text":
            image_path = os.path.join(image_root, item["image"])
            if args.answer_directly:
                prompt = item["question"] + " Directly answer the question using a single word or phrase without explanation."
            else:
                prompt = item["question"] + " The last line of your response should be of the following format: 'Answer: YOUR_ANSWER' where YOUR_ANSWER is the final answer. Think step by step before answering."
        else:
            print("Unknown input modal.")
            raise ValueError
        
        messages = [
            {
                "role": "user", 
                "content": [
                    {"type": "image", "image": Image.open(image_path).convert("RGB")},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        input_ids, pixel_values, grid_thws = model.preprocess_inputs(
            messages=messages,
            add_generation_prompt=True,
            enable_thinking=False
        )
        input_ids = input_ids.to(model.device)
        pixel_values = pixel_values.to(model.device) if pixel_values is not None else None
        grid_thws = grid_thws.to(model.device) if grid_thws is not None else None
        outputs = model.generate(
            **generate_kwargs,
            inputs=input_ids,
            pixel_values=pixel_values,
            grid_thws=grid_thws,
            enable_thinking=False
        )
        response = model.text_tokenizer.decode(outputs[0], skip_special_tokens=True)
        item["response"] = response
        
        if verbose:
            print(f'Prediction: {response}\nGT: {item["solution"]}')
        
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
    
    log_path = os.path.join(args.output_folder, model_name + f'_{args.lmm_input_modal}_{answer_mode}.json')
    save_json(data, log_path)
    print(f"save the predictions to {log_path}")