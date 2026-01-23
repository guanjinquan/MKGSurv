import json
import random


file_pair = "/home/Guanjq/NewWork/MedAlignFusion/Data/TCGA-BRCA/processed/pairs_knowledge_qwen.json"
file_kimi = "/home/Guanjq/NewWork/MedAlignFusion/Data/TCGA-BRCA/processed/medical_analysis_kimi.json"


with open(file_pair, "r") as f:
    pair_knowledge = json.load(f)

with open(file_kimi, "r") as f:
    kimi_knowledge = json.load(f)

for k, v in kimi_knowledge.items():
    if k not in pair_knowledge:
        continue
    kimi_list = v
    know_list = pair_knowledge[k]
    for i in range(len(know_list)):
        for j in range(len(kimi_list)):
            if know_list[i]['modalPairs'][0] == kimi_list[j]['modalPairs'][0] and know_list[i]['modalPairs'][1] == kimi_list[j]['modalPairs'][1]:
                short_knowledge = know_list[i]['knowledge']
                sent_list = short_knowledge.split(". ")
                short_knowledge = ". ".join(sent_list[:max(5, random.randint(1, len(sent_list)))])
                kimi_list[j]['survival'] = short_knowledge + ". " + kimi_list[j]['survival']


with open(file_kimi, "w") as f:
    json.dump(kimi_knowledge, f)