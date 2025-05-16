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
import pandas as pd


interleaved_prompt = (
    "Generate an image about random color paper with smallest font-size. Firstly, it is written about context information: \"{context}\"\n"
    "An illustration figure or scene described by the above context is shown.\n"
    "Then, the image displays a question: \"{question}\"\n"
    "Finally, four choices are written on the image: \"{choices}\"\n"
    "Do not summarize the visual text content given above."
    )

interleaved_prompt_handwritten = (
    "Generate an image about random color paper with smallest font-size. Firstly, it is written about context information in handwritten style: \"{context}\"\n"
    "An illustration figure or scene described by the above context is shown.\n"
    "Then, the image displays a question in handwritten style: \"{question}\"\n"
    "Finally, four choices in handwritten style are written on the image: \"{choices}\"\n"
    "Do not summarize the visual text content given above."
    )

with_background_prompt = (
    "Generate an image with smallest font-size. {caption}\n"
    "Some text paragraphs with contrastive color are shown. "
    "Specifically, firstly, it is written about context information: \"{context}\"\n"
    "Then, the image displays a question: \"{question}\"\n"
    "Finally, four choices are written on the image: \"{choices}\"\n"
    "Do not summarize the visual text content given above."
    )

with_background_prompt_handwritten = (
    "Generate an image with smallest font-size. {caption}\n"
    "Some text paragraphs with handwritten style and contrastive color are shown. "
    "Specifically, firstly, it is written about context information: \"{context}\"\n"
    "Then, the image displays a question: \"{question}\"\n"
    "Finally, four choices are written on the image: \"{choices}\"\n"
    "Do not summarize the visual text content given above."
    )


def gen_image(data, output_folder):
    retry_count = 2
    retry_interval = 1
    
    for _ in range(retry_count):
        try:
            client = OpenAI(api_key=openai_key, base_url=base_url)
            if data["caption_or_not"] == "yes":
                if data["handwritten"] == "yes":
                    gpt_prompt = with_background_prompt_handwritten.format(
                        caption=data["caption"],
                        context=data["context"],
                        question=data["question"],
                        choices=data["choices"],
                        )
                else:
                    gpt_prompt = with_background_prompt.format(
                        caption=data["caption"],
                        context=data["context"],
                        question=data["question"],
                        choices=data["choices"],
                        )
                gen_size = random.choice(["1536x1024", "1024x1536"])
            else:
                if data["handwritten"] == "yes":
                    gpt_prompt = interleaved_prompt_handwritten.format(
                        context=data["context"],
                        question=data["question"],
                        choices=data["choices"],
                        )
                else:
                    gpt_prompt = interleaved_prompt.format(
                        context=data["context"],
                        question=data["question"],
                        choices=data["choices"],
                        )
                if data["tokens"] >= 160:
                    gen_size = "1536x1024"
                else:
                    gen_size = random.choice(["1536x1024", "1024x1536"])
            
            result = client.images.generate(
            model="gpt-image-1",
            prompt=gpt_prompt,
            size=gen_size,
            quality="high",
        )
            image_bytes = base64.b64decode(result.data[0].b64_json)  # token_usage = result.usage
            save_name = os.path.join("saved_folder", str(data["id"])+'.jpg')
            with open(save_name, "wb") as f:
                f.write(image_bytes)
            
            print(f'id: {data["id"]} Success! Saved to {save_name}\n')
            print(gpt_prompt)
            print("\n\n")
            return data["id"], "success"
        
        except Exception as e:
            print("ID: ", data["id"], " Error: ", e)
            print("Request again...")
            time.sleep(retry_interval)
        
    return data["id"], "failed"


if __name__ == "__main__":
    read_path = sys.argv[1]
    openai_key = sys.argv[2]
    base_url = sys.argv[3]
    num_workers = int(sys.argv[4])
    output_folder = "saved_folder"
    os.makedirs(output_folder, exist_ok=True)
    
    with open(read_path, 'r') as f:
        data_list = json.load(f)
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(gen_image, data) for data in data_list]
        id2res = collections.defaultdict()
        for job in as_completed(futures):
            data_id, res = job.result(timeout=None)
            id2res[str(data_id)] = res
            time.sleep(1)

    df = pd.DataFrame(
        {
            'id': list(id2res.keys()),
            'result': list(id2res.values())
        }
        )
    df.to_excel('./gen_history.xlsx', index=False)
    
    