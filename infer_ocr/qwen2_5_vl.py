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
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--image_folder", type=str, default="./images")
    parser.add_argument("--json_file", type=str, default="LogicOCR.json")
    parser.add_argument("--output_folder", type=str, default="./res_ocr")
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
            max_new_tokens=512,
            do_sample=False,
        )

    for item in tqdm(data):
        image_path = os.path.join(image_root, item["image"])
        messages = [
            {
                "role": "user", 
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": "Please recognize the text paragraphs in the image. Do not add explanation. Do not answer the question in this image."}
                ]
            }
        ]
        
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
        if verbose:
           print(response) 
        item["ocr_res"] = response
        
    output_queue.put({eval_id: data})
    print(f"Process {eval_id} has completed.")
        

if __name__ == "__main__":
    args = _get_args()
    model_name = args.model_path.split('/')[-1]
    print(f"OCR with: {model_name}")
    
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
    
    log_path = os.path.join(args.output_folder, model_name + f'_ocr.json')
    save_json(data, log_path)
    print(f"save the predictions to {log_path}")