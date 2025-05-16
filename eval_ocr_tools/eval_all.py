import json
import argparse
import nltk
from nltk.metrics import precision, recall, f_measure
import numpy as np
import jieba
import re
from nltk.translate import meteor_score
from tqdm import tqdm
from rouge import Rouge
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

def contain_chinese_string(text):
    chinese_pattern = re.compile(r'[\u4e00-\u9fa5]')
    return bool(chinese_pattern.search(text))

def cal_per_metrics(pred, gt):

    metrics = {}

    if contain_chinese_string(gt) or contain_chinese_string(pred):
        reference = jieba.lcut(gt)
        hypothesis = jieba.lcut(pred)
    else:
        reference = gt.split()
        hypothesis = pred.split()

    metrics["bleu"] = nltk.translate.bleu([reference], hypothesis)
    metrics["meteor"] = meteor_score.meteor_score([reference], hypothesis)

    rouge_score = Rouge().get_scores(hyps=' '.join(hypothesis), refs=' '.join(reference))
    metrics["rouge-l f"] = rouge_score[0]['rouge-l']['f']
    
    reference = set(reference)
    hypothesis = set(hypothesis)
    metrics["f_measure"] = f_measure(reference, hypothesis)
    metrics["precision"] = precision(reference, hypothesis)
    metrics["recall"] = recall(reference, hypothesis)
    metrics["edit_dist"] = nltk.edit_distance(pred, gt) / max(len(pred), len(gt))
    
    return metrics

def eval_worker(data_chunk, eval_id, output_queue):
    result = []
    for ann in tqdm(data_chunk):
        ann_gt = ' '.join([ann['context'], ann['question'], ann['choices'].replace('\n', ' ')])
        ann_gt = ann_gt.lower()
        ann_pred = ann['ocr_res'].replace('\n\n', ' ').replace('\n', ' ')
        ann_pred = ann_pred.lower()
        ans = cal_per_metrics(ann_pred, ann_gt)
        result.append(ans)
    output_queue.put({eval_id: result})
    print(f"Process {eval_id} has completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()
    
    multiprocessing.set_start_method('spawn')
    
    with open(args.input_file, 'r') as f:
        data = json.load(f)
    data_list = split_list(data, args.num_workers)
    
    output_queue = Manager().Queue()
    pool = Pool(processes=args.num_workers)
    for i in range(len(data_list)):
        pool.apply_async(
            eval_worker, 
            args=(data_list[i], i, output_queue)
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
    
    mean_dict = {}
    mean_dict["eval question num"] = len(data)
    for k, v in data[0].items():
        mean_dict[k] = 0
    
    for each in data:
        for k, v in each.items():
            mean_dict[k] += v

    for k, v in mean_dict.items():
        if k == "eval question num":
            continue
        mean_dict[k] /= len(data)
    
    print(json.dumps(mean_dict, indent=4))
    with open(args.output_file, 'w') as fw:
        json.dump(mean_dict, fw, indent=4)
    print(f'results saved to {args.output_file}')