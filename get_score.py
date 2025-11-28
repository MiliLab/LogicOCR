import sys
import json
from argparse import ArgumentParser
from tqdm import tqdm
import os


def _get_args():
    parser = ArgumentParser()
    parser.add_argument("--gen_json", type=str, required=True)
    parser.add_argument("--real_json", type=str, required=True)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = _get_args()
    
    total_score = 0
    total_ques = 0
    
    with open(args.gen_json, 'r') as f:
        gen_res = json.load(f)
    assert len(gen_res) <= 1100  # QvQ API may reject some questions
    
    gen_score_by_rtype = {
        "1 reasoning type": 0, 
        "2 reasoning type": 0, 
        "3 reasoning type": 0, 
        ">3 reasoning type": 0, 
    }
    gen_ques_by_rtype = {
        "1 reasoning type": 0, 
        "2 reasoning type": 0, 
        "3 reasoning type": 0, 
        ">3 reasoning type": 0, 
    }
    for item in gen_res:
        if "score" not in item.keys():
            continue
        total_ques += 1
        
        if len(item["type"].keys()) <= 1:
            gen_ques_by_rtype["1 reasoning type"] += 1
        elif len(item["type"].keys()) == 2:
            gen_ques_by_rtype["2 reasoning type"] += 1
        elif len(item["type"].keys()) == 3:
            gen_ques_by_rtype["3 reasoning type"] += 1
        elif len(item["type"].keys()) > 3:
             gen_ques_by_rtype[">3 reasoning type"] += 1
             
        if item["score"]:
            total_score += 1
            if len(item["type"].keys()) <= 1:
                gen_score_by_rtype["1 reasoning type"] += 1
            elif len(item["type"].keys()) == 2:
                gen_score_by_rtype["2 reasoning type"] += 1
            elif len(item["type"].keys()) == 3:
                gen_score_by_rtype["3 reasoning type"] += 1
            elif len(item["type"].keys()) > 3:
                gen_score_by_rtype[">3 reasoning type"] += 1
    assert sum(gen_ques_by_rtype.values()) <= len(gen_res)
    
    print(f"LogicOCR_Gen accuracy: {round(sum(gen_score_by_rtype.values())/sum(gen_ques_by_rtype.values())*100, 3)}")
    for key, value in gen_score_by_rtype.items():
        gen_score_by_rtype[key] = round(value / gen_ques_by_rtype[key] * 100, 3)
    print(gen_score_by_rtype)
    print('-'*30)
    
    
    
    with open(args.real_json, 'r') as f:
        real_res = json.load(f)
    real_score_by_rtype = {
        "numerical": 0, 
        "temporal": 0, 
        "decision": 0, 
        "conditional": 0, 
    }
    real_ques_by_rtype = {
        "numerical": 0, 
        "temporal": 0, 
        "decision": 0, 
        "conditional": 0, 
    }
    for item in real_res:
        if "evaluation" not in item.keys():
            continue
        total_ques += 1
        
        if item["type"] in ["data comparison analysis", "data statistical analysis", "mathematical reasoning"]:
            real_ques_by_rtype["numerical"] += 1
        elif item["type"] == "temporal reasoning":
            real_ques_by_rtype["temporal"] += 1
        elif item["type"] == "decision reasoning":
            real_ques_by_rtype["decision"] += 1
        elif item["type"] == "conditional reasoning":
            real_ques_by_rtype["conditional"] += 1
        else:
            raise ValueError
        
        if item["evaluation"]["correct"]:
            total_score += 1
            if item["type"] in ["data comparison analysis", "data statistical analysis", "mathematical reasoning"]:
                real_score_by_rtype["numerical"] += 1
            elif item["type"] == "temporal reasoning":
                real_score_by_rtype["temporal"] += 1
            elif item["type"] == "decision reasoning":
                real_score_by_rtype["decision"] += 1
            elif item["type"] == "conditional reasoning":
                real_score_by_rtype["conditional"] += 1
            else:
                raise ValueError
    # assert sum(real_ques_by_rtype.values()) == len(real_res)
    
    print(f"LogicOCR_Real accuracy: {round(sum(real_score_by_rtype.values())/sum(real_ques_by_rtype.values())*100, 3)}")
    for key, value in real_score_by_rtype.items():
        real_score_by_rtype[key] = round(value / real_ques_by_rtype[key] * 100, 3)
    print(real_score_by_rtype)
    print('-'*30)
    
    print(f"{total_ques} questions.\naccuracy: {round(total_score/total_ques*100, 3)}")