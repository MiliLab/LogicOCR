import sys
import json
from argparse import ArgumentParser
from tqdm import tqdm
import re
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
import torch
import multiprocessing
from multiprocessing import Pool, Queue, Manager
try:
    from qwen_vl_utils import process_vision_info
except Exception as err:
    print("qwen_vl_utils not found, please install it via 'pip install qwen-vl-utils'")
    raise err
import os


DIRECT_PROMPT = "Directly answer the question with one option letter without explanation."
CoT_PROMPT = "Solve the above multiple-choice question and then answer with one option letter. The last line of your response should be of the following format: 'Answer: LETTER' where LETTER is one of options. Think step by step before answering."


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


def parse_response(response, ans):
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
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--json_file", type=str, default="LogicOCR.json")
    parser.add_argument("--output_folder", type=str, default="./res_ocr")
    parser.add_argument("--lmm_input_modal", type=str, default="text")
    parser.add_argument("--answer_directly", action='store_true')
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("--auto_device", action='store_true')
    args = parser.parse_args()
    return args


def infer_worker(args, data, eval_id, output_queue):
    device =  f"cuda:{eval_id}"
    if args.auto_device:
        device = "auto"
    input_modal = args.lmm_input_modal
    model_path = args.model_path
    verbose = args.verbose
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            attn_implementation="flash_attention_2",
        ).eval()
    processor = AutoProcessor.from_pretrained(model_path)
    generate_kwargs = dict(
            max_new_tokens=2048,
            do_sample=False,
        )

    for item in tqdm(data):
        if input_modal == "text":
            problem = item['ocr_res'].replace("```plaintext\n", "").replace("\n```", "")  # Qwen2.5-VL-32B tend to add prefix and suffix
            problem = problem.replace("\n\n", "\n")
            if args.answer_directly:
                prompt = problem + '\n\n' + DIRECT_PROMPT
            else:
                prompt = problem + '\n\n' + CoT_PROMPT
            messages = [
                {"role": "user", "content": [{"type": "text", "text": prompt}]}
            ]
        else:
            print("Unknown input modal.")
            raise ValueError
        
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            )

        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            ).to(model.device)

        generated_ids = model.generate(**inputs, **generate_kwargs)
        generated_ids = [
            out_ids[inputs.input_ids.shape[1]:] for out_ids in generated_ids
        ]
        response = processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
        item["response"] = response
        
        score = 0
        response_choice = parse_response(response, None)
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
    
    scores = [dd["score"] for dd in data]
    print(f"Average score: {sum(scores)/len(scores)}")