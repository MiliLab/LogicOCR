import sys
import json
from argparse import ArgumentParser
from tqdm import tqdm
import re
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, AutoProcessor
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
import multiprocessing
from multiprocessing import Pool, Queue, Manager
import os
try:
    import deepseek_vl2
except Exception as e:
    print('Please first install deepseek_vl2 from source codes in: https://github.com/deepseek-ai/DeepSeek-VL2')
    raise e


DIRECT_PROMPT = "Directly answer the question with one option letter without explanation."
DIRECT_MM_PROMPT = "Solve the multiple-choice question in image. Directly answer the question with one option letter without explanation."
CoT_PROMPT = "Solve the multiple-choice question and then answer with one option letter. The last line of your response should be of the following format: 'Answer: LETTER' where LETTER is one of options. Think step by step before answering."
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
    return None

            
def _get_args():
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, default="deepseek-ai/deepseek-vl2")
    parser.add_argument("--image_folder", type=str, default="./images")
    parser.add_argument("--json_file", type=str, default="LogicOCR.json")
    parser.add_argument("--output_folder", type=str, default="./res")
    parser.add_argument("--lmm_input_modal", type=str, default="image-text")  # "text"
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
    image_root = args.image_folder
    input_modal = args.lmm_input_modal
    model_path = args.model_path
    verbose = args.verbose
    
    from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
    from deepseek_vl2.utils.io import load_pil_images
    vl_chat_processor = DeepseekVLV2Processor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer
    model: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
        ).eval()
    
    generate_kwargs = dict(
            max_new_tokens=2048,
            do_sample=False,
            use_cache=True
        )

    for item in tqdm(data):
        
        if input_modal == "image-text":
            image_path = os.path.join(image_root, item["image"])
            if args.answer_directly:
                prompt = DIRECT_MM_PROMPT
            else:
                prompt = CoT_MM_PROMPT
            conversation = [
                {
                    "role": "<|User|>",
                    "content": f"<image>\n{prompt}",
                    "images": [image_path],
                },
                {"role": "<|Assistant|>", "content": ""},
            ]
            pil_images = load_pil_images(conversation)
            prepare_inputs = vl_chat_processor(
                conversations=conversation,
                images=pil_images,
                force_batchify=True,
                system_prompt=""
            ).to(model.device)
            with torch.no_grad():
                inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)
                inputs_embeds, past_key_values = model.incremental_prefilling(
                    input_ids=prepare_inputs.input_ids,
                    images=prepare_inputs.images,
                    images_seq_mask=prepare_inputs.images_seq_mask,
                    images_spatial_crop=prepare_inputs.images_spatial_crop,
                    attention_mask=prepare_inputs.attention_mask,
                    chunk_size=512
                )
                outputs = model.generate(
                    inputs_embeds=inputs_embeds,
                    input_ids=prepare_inputs.input_ids,
                    images=prepare_inputs.images,
                    images_seq_mask=prepare_inputs.images_seq_mask,
                    images_spatial_crop=prepare_inputs.images_spatial_crop,
                    attention_mask=prepare_inputs.attention_mask,
                    past_key_values=past_key_values,
                    pad_token_id=tokenizer.eos_token_id,
                    bos_token_id=tokenizer.bos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    **generate_kwargs
                )
                response = tokenizer.decode(
                    outputs[0][len(prepare_inputs.input_ids[0]):].cpu().tolist(),
                    skip_special_tokens=True
                )
                response = response.rstrip('.')
            
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