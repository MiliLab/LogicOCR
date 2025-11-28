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
import random
import copy
random.seed(1234)
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


def save_json(json_list, save_path):
    with open(save_path, 'w') as file:
        json.dump(json_list, file, indent=4)

       
def _get_args():
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Efficient-Large-Model/NVILA-8B")
    parser.add_argument("--image_folder", type=str, default="./LogicOCR_real")
    parser.add_argument("--json_file", type=str, default="LogicOCR_real.json")
    parser.add_argument("--output_folder", type=str, default="./res_real")
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
            query = instance["question"] + " Directly answer the question using a single word or phrase without explanation."
        else:
            query = instance["question"] + " The last line of your response should be of the following format: 'Answer: YOUR_ANSWER' where YOUR_ANSWER is the final answer. Think step by step before answering."
        response = model.generate_content([image, query], generation_config=generation_config)
        output = copy.deepcopy(instance)
        output["response"] = response
        if args.verbose:
            print(f'Prediction: {response}\nGT: {instance["solution"]}')
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
