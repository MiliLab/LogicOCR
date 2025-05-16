import sys
import json
from argparse import ArgumentParser
from tqdm import tqdm
import re
from transformers import AutoModel, AutoTokenizer
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
import multiprocessing
from multiprocessing import Pool, Queue, Manager
import os


DIRECT_PROMPT = "Directly answer the question with one option letter without explanation."
DIRECT_MM_PROMPT = "Solve the multiple-choice question in image. Directly answer the question with one option letter without explanation."
CoT_PROMPT = "Solve the multiple-choice question and then answer with one option letter. The last line of your response should be of the following format: 'Answer: LETTER' where LETTER is one of options. Think step by step before answering."
CoT_MM_PROMPT = "Solve the multiple-choice question in image and then answer with one option letter. The last line of your response should be of the following format: 'Answer: LETTER' where LETTER is one of options. Think step by step before answering."

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images, target_aspect_ratio


def dynamic_preprocess2(image, min_num=1, max_num=12, prior_aspect_ratio=None, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    new_target_ratios = []
    for i in target_ratios:
        if prior_aspect_ratio[0]%i[0] or prior_aspect_ratio[1]%i[1]:
            new_target_ratios.append(i)
        else:
            continue
    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, new_target_ratios, orig_width, orig_height, image_size)
    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, min_num=1, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images, target_aspect_ratio = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, min_num=min_num, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values, target_aspect_ratio


def load_image2(image_file, input_size=448, min_num=1, max_num=12, target_aspect_ratio=None):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess2(image, image_size=input_size, use_thumbnail=True, min_num=min_num, max_num=max_num, prior_aspect_ratio=target_aspect_ratio)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


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
    parser.add_argument("--model_path", type=str, default="mx262/MiniMonkey")
    parser.add_argument("--image_folder", type=str, default="./images")
    parser.add_argument("--json_file", type=str, default="LogicOCR.json")
    parser.add_argument("--output_folder", type=str, default="./res")
    parser.add_argument("--lmm_input_modal", type=str, default="image-text")  # or "text"
    parser.add_argument("--answer_directly", action='store_true')
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--verbose", action='store_true')
    args = parser.parse_args()
    return args


def infer_worker(args, data, eval_id, output_queue):
    device =  f"cuda:{eval_id}"
    image_root = args.image_folder
    input_modal = args.lmm_input_modal
    model_path = args.model_path
    verbose = args.verbose
    
    model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            attn_implementation="flash_attention_2",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    generate_kwargs = dict(
            max_new_tokens=2048,
            do_sample=False,
        )

    for item in tqdm(data):
        
        if input_modal == "text":
            problem = 'Question: '+item["context"].strip()+' '+item["question"].strip()+'\nOptions:\n'+item["choices"]
            if args.answer_directly:
                prompt = problem + '\n' + DIRECT_PROMPT
            else:
                prompt = problem + '\n' + CoT_PROMPT
            response = model.chat(tokenizer, None, prompt, generate_kwargs)
                
        elif input_modal == "image-text":
            image_path = os.path.join(image_root, item["image"])
            pixel_values, target_aspect_ratio = load_image(image_path, min_num=4, max_num=12)
            pixel_values = pixel_values.to(torch.bfloat16).to(model.device)
            pixel_values2 = load_image2(image_path, min_num=3, max_num=7, target_aspect_ratio=target_aspect_ratio)
            pixel_values2 = pixel_values2.to(torch.bfloat16).to(model.device)
            pixel_values = torch.cat([pixel_values2[:-1], pixel_values[:-1], pixel_values2[-1:]], 0)
            if args.answer_directly:
                prompt = DIRECT_MM_PROMPT
            else:
                prompt = CoT_MM_PROMPT
            response, history = model.chat(tokenizer, pixel_values, target_aspect_ratio, prompt, generate_kwargs, history=None, return_history=True)
        
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