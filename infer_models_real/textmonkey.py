import sys
import json
from argparse import ArgumentParser
from tqdm import tqdm
import codecs
import re
import os

import random
random.seed(1234)
custom_transformers_path = "/mnt/whuscs/hhb/vlm/VLMEvalKit/reasoning_qa/transformer_4_32"
sys.path.insert(0, custom_transformers_path)
sys.path.append("/mnt/whuscs/hhb/vlm/VLMEvalKit/reasoning_qa/Monkey/")
from eval.vqa import VQA
from eval.vqa_eval import VQAEval
from monkey_model.modeling_textmonkey import TextMonkeyLMHeadModel
from monkey_model.tokenization_qwen import QWenTokenizer
from monkey_model.configuration_qwen import QWenConfig
from monkey_model.configuration_monkey import MonkeyConfig
import multiprocessing
from multiprocessing import Pool, Queue, Manager


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
    parser.add_argument("--model_path", type=str, default="")
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
    
    config = MonkeyConfig.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    model = TextMonkeyLMHeadModel.from_pretrained(
        model_path,
        config=config,
        device_map=device, 
        trust_remote_code=True
        ).eval()
    tokenizer = QWenTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token_id = tokenizer.eod_id
    tokenizer.IMG_TOKEN_SPAN = config.visual["n_queries"]
    

    for item in tqdm(data):
        if input_modal == "image-text":
            image_path = os.path.join(image_root, item["image"])
            if args.answer_directly:
                prompt = item["question"] + " Directly answer the question using a single word or phrase without explanation."
            else:
                prompt = item["question"] + " The last line of your response should be of the following format: 'Answer: YOUR_ANSWER' where YOUR_ANSWER is the final answer. Think step by step before answering."
            message = [f'<img>{image_path}</img> {prompt}']
            input_ids = tokenizer(message, return_tensors='pt', padding='longest').to(model.device)
            content = model.generate(
                    input_ids=input_ids.input_ids,
                    attention_mask=input_ids.attention_mask,
                    do_sample=False,
                    temperature=0,
                    num_beams=1,
                    max_new_tokens=2048,
                    min_new_tokens=1,
                    length_penalty=1,
                    num_return_sequences=1,
                    output_hidden_states=False,
                    use_cache=True,
                    pad_token_id=tokenizer.eod_id,
                    eos_token_id=tokenizer.eod_id,
                )
            response = tokenizer.decode(content[0][input_ids.input_ids.size(1):].cpu(), skip_special_tokens=True).strip().replace("<|endoftext|>","")
            response = response.split('(')[0]
        else:
            print("Unknown input modal.")
            raise ValueError
        
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