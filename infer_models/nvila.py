import sys
import json
from argparse import ArgumentParser
from tqdm import tqdm
import re
import torch
from PIL import Image
import multiprocessing
from multiprocessing import Pool, Queue, Manager
import os
import itertools
import copy
try:
    import llava
    from llava import conversation as conversation_lib
    from llava.utils import distributed as dist
    from llava.utils import io
    from llava.utils.logging import logger
except Exception as err:
    print('Please install VILA before using VILA')
    print('Please install VILA from https://github.com/NVlabs/VILA')
    print('Please install VLMEvalKit after installing VILA')
    print('VILA is supported only with transformers==4.36.2')
    raise err


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
    parser.add_argument("--model_path", type=str, default="Efficient-Large-Model/NVILA-8B")
    parser.add_argument("--image_folder", type=str, default="./images")
    parser.add_argument("--json_file", type=str, default="LogicOCR.json")
    parser.add_argument("--output_folder", type=str, default="./res")
    parser.add_argument("--lmm_input_modal", type=str, default="image-text")  # "text"
    parser.add_argument("--answer_directly", action='store_true')
    parser.add_argument("--verbose", action='store_true')
    args = parser.parse_args()
    return args


def main(args, answer_mode) -> None:
    model_name = args.model_path.split('/')[-1]
    # Set up distributed environment
    dist.init()
    devices = range(dist.local_rank(), torch.cuda.device_count(), dist.local_size())
    torch.cuda.set_device(devices[0])

    conversation_lib.default_conversation = conversation_lib.conv_templates["auto"].copy()

    # Load model
    model = llava.load(args.model_path, devices=devices)
    model.config.min_tiles = 1
    model.config.max_tiles = 9
    model.llm.config.min_tiles = 1
    model.llm.config.max_tiles = 9

    # if args.max_tiles > 12:
    #     context_length = int(args.max_tiles / 12.0 * 4096)
    #     model.config.model_max_length = context_length
    #     model.config.tokenizer_model_max_length = context_length
    #     model.llm.config.model_max_length = context_length
    #     model.llm.config.tokenizer_model_max_length = context_length

    # Set up generation config
    generation_config = model.default_generation_config
    generation_config.update(max_new_tokens=2048)

    # Load data and chunk it
    instances = io.load(args.json_file)[dist.rank() :: dist.size()]

    # Run inference
    outputs = []
    for instance in tqdm(instances):
        image = Image.open(os.path.join(args.image_folder, instance["image"]))
        if args.answer_directly:
            query = DIRECT_MM_PROMPT
        else:
            query = CoT_MM_PROMPT
        response = model.generate_content([image, query], generation_config=generation_config)
        output = copy.deepcopy(instance)
        output["response"] = response
        score = 0
        response_choice = parse_response(response, instance["choices"])
        if response_choice!=None and response_choice.lower() == instance["solution"].lower():
            score = 1
        if args.verbose:
            print(f'{response}\nPrediction: {response_choice}, GT: {instance["solution"]}, Score: {score}')
        output["score"] = score
        outputs.append(output)

    # Gather and save outputs
    if dist.size() > 1:
        outputs = dist.gather(outputs, dst=0)
        if not dist.is_main():
            return
        outputs = list(itertools.chain(*outputs))
    
    log_path = os.path.join(args.output_folder, model_name+f'_{args.lmm_input_modal}_{answer_mode}.json')
    io.save(log_path, outputs)
    print(f"save the predictions to {log_path}")
    
    scores = [dd["score"] for dd in outputs]
    print(f"Average score: {sum(scores)/len(scores)}, {len(scores)} samples")


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
    os.makedirs(args.output_folder, exist_ok=True)
    main(args, answer_mode)
