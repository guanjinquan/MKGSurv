import json
import os
from collections import defaultdict
import numpy as np

data_type_mapping = {
    # Clinical 相关
    'clinical data': 'clinical',
    'clinical': 'clinical',
    
    # Treatment 相关
    'treatment data': 'treatment',
    'treatment': 'treatment',
    
    # Pathology 相关
    'pathological data': 'pathology',
    'pathological': 'pathology',
    'pathology': 'pathology',
    
    # Genomics 相关
    'genomic data': 'genomics',
    'genomics data': 'genomics',
    'genomic': 'genomics',
    'genomics': 'genomics',
    'genomics': 'genomics',
    'molecular data': 'genomics'
}

def map_data_type(data_string):
    """
    将数据类型的字符串描述映射到四个标准关键字之一
    """
    # 转换为小写并去除前后空格
    normalized = data_string.lower().strip()
    normalized = normalized.replace("data","").strip()

    # 直接查找映射
    if normalized in data_type_mapping:
        return data_type_mapping[normalized]
    
    # 如果直接查找失败，尝试包含性匹配
    if 'clinical' in normalized:
        return 'clinical'
    elif 'treatment' in normalized:
        return 'treatment'
    elif 'pathology' in normalized or 'pathological' in normalized:
        return 'pathology'
    elif 'genomic' in normalized or 'genomics' in normalized or 'molecular' in normalized:
        return 'genomics'
    else:
        # 默认返回原字符串的小写形式
        return normalized

# file = "/home/Guanjq/NewWork/MedAlignFusion/Data/TCGA-BRCA/processed/medical_analysis_deepseek.json"
file = "/home/Guanjq/NewWork/MedAlignFusion/Data/TCGA-BRCA/processed/medical_analysis_qwen.json"


with open(file, 'r') as f:
    data = json.load(f)

scores = defaultdict(list)

for k, v in data.items():
    assert len(v) == 6
    for d in v:
        d['modalPairs'].sort()
        d['modalPairs'] = [map_data_type(s) for s in d['modalPairs']]
        try:
            pair = (d['modalPairs'][0], d['modalPairs'][1])
            scores[pair].append(d['score'])
        except:
            print(k, d)

for k, v in scores.items():
    print(k, np.mean(v), np.std(v))