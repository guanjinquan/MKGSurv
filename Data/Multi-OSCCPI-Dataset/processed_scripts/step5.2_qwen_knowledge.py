import pandas as pd
import json
import os
import time
import threading
import re
from itertools import combinations
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ================= CONFIGURATION =================
API_KEY = os.environ.get("QWEN_API_KEY", "") 
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "qwen-turbo"

# File Paths
INPUT_CSV_PATH = "/home/Guanjq/NewWork/MedAlignFusion/Data/Multi-OSCCPI-Dataset/processed/multimodal_texts.csv"
OUTPUT_JSON_PATH = "/home/Guanjq/NewWork/MedAlignFusion/Data/Multi-OSCCPI-Dataset/processed/pairs_knowledge_qwen.json"
FAILED_LOG_PATH = "/home/Guanjq/NewWork/MedAlignFusion/Data/Multi-OSCCPI-Dataset/failed_ids_lusc_qwen.log"

# Threading Configuration
MAX_WORKERS = 5  # 保持适度的并发，避免触发限流

# Modalities to pair
MODALITIES_KEYS = ['clinical', 'pathology', 'treatment']

# ================= CLIENT SETUP =================
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def construct_prompt(source1, data1, source2, data2):
    """
    Constructs the prompt string with the specific patient data using the user's template.
    """    
    prompt = f"""
You are a medical expert.
Please analyze the following patient data from two different departments:

**{source1}**: {data1}

**{source2}**: {data2}

**Task:**
Synthesize all provided data sources to generate a single, dense, and professional analytical paragraph. Your analysis must strictly follow these steps:

1.  **Cross-Departmental Correlation:**
    * Analyze the consistency and discrepancies between the different data sources.
    * Identify if specific biomarkers or signals from one department amplify or mitigate the risks suggested by another department.

2.  **Survival Analysis:**
    * Apply tumor-specific medical reasoning to determine the patient's survival outlook.
    * Weigh favorable factors (e.g., early stage, negative margins) against adverse features (e.g., aggressive pathways, high proliferation indices).

3.  **Risk Quantification:**
    * Based on the integrative analysis, conclude with a specific **Risk Score** on a scale of **0 to 10**.
    * **0** = Negligible Risk / Cured / Excellent Prognosis.
    * **10** = Extreme Risk / Imminent Mortality / Poor Prognosis.

**Output Format:**
<output> Provide one comprehensive paragraph containing the medical analysis. </output>

Stop after generating above output. 
"""
    return prompt

def extract_response_text(response_text):
    """
    Extracts text between <output> tags. 
    If tags are missing, returns the whole trimmed text.
    """
    if not response_text:
        return ""
    
    text = response_text.strip()
    
    # 简单的字符串查找，比正则容错率更高
    start_tag = "<output>"
    end_tag = "</output>"
    
    start_idx = text.find(start_tag)
    end_idx = text.rfind(end_tag)
    
    if start_idx != -1 and end_idx != -1:
        # 提取标签中间的内容
        return text[start_idx + len(start_tag) : end_idx].strip()
    elif start_idx != -1:
        # 只有开始标签，提取到最后
        return text[start_idx + len(start_tag):].strip()
    else:
        # 没有标签，直接返回全文
        # 这里可以选择去掉可能的 markdown 代码块符号
        text = text.replace("```html", "").replace("```xml", "").replace("```", "")
        return text.strip()

# ================= WORKER FUNCTION =================

def process_single_patient(row_data, final_data_ref, file_lock):
    """
    Worker function to process a single patient.
    Generates C(4,2) pairs and calls API for each valid pair.
    """
    submitter_id = row_data['PID']
    patient_results = []
    
    # 1. 准备有效数据字典
    valid_data_map = {}
    for key in MODALITIES_KEYS:
        content = row_data.get(key)
        # 过滤无效数据：None, NaN, "Not Available", 空字符串
        if content and str(content).strip().lower() not in ["not available", "nan", "none", ""]:
            valid_data_map[key] = str(content)
            
    # 2. 生成所有两两组合
    # C(4, 2) = 6 pairs (clinical-pathology, clinical-treatment, etc.)
    pairs = list(combinations(MODALITIES_KEYS, 2))
    
    for mod1, mod2 in pairs:
        # 只有当两个模态都有数据时才处理
        if mod1 not in valid_data_map or mod2 not in valid_data_map:
            continue
            
        data1 = valid_data_map[mod1]
        data2 = valid_data_map[mod2]
        
        prompt_text = construct_prompt(mod1, data1, mod2, data2)
        
        # 3. API 调用 (带重试)
        pair_success = False
        MAX_RETRIES = 3 
        
        for attempt in range(MAX_RETRIES):
            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        # System prompt 可以简单一点，让模型专注于 User Prompt 的指令
                        {'role': 'system', 'content': 'You are a helpful medical assistant.'},
                        {'role': 'user', 'content': prompt_text}
                    ],
                    temperature=0.3, # 稍微降低随机性，保证分析稳定性
                    timeout=60
                )
                
                raw_response = completion.choices[0].message.content
                extracted_content = extract_response_text(raw_response)
                
                # 记录结果
                result_entry = {
                    "modalPairs": [mod1, mod2],
                    "knowledge": extracted_content
                }
                
                patient_results.append(result_entry)
                pair_success = True
                break 
                
            except Exception as e:
                # 发生错误暂停一下再重试
                time.sleep(1 + attempt)
        
        if not pair_success:
            print(f"Warning: Failed to process pair {mod1}-{mod2} for {submitter_id}")

    # 如果没有任何一对成功生成（且本来有数据），则视为该病人处理失败
    if not patient_results and len(valid_data_map) >= 2:
        return False, submitter_id, "API failed for all valid pairs"
    
    # 如果本来数据就不够两两配对，也算成功（只是结果为空列表）
    
    # 4. 原子写入文件
    try:
        with file_lock:
            # 更新内存中的大字典
            final_data_ref[submitter_id] = patient_results
            
            # 写入临时文件并重命名
            temp_file = OUTPUT_JSON_PATH + ".tmp"
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(final_data_ref, f, ensure_ascii=False, indent=4)
            
            os.replace(temp_file, OUTPUT_JSON_PATH)
            
        return True, submitter_id, "Success"
        
    except Exception as e:
        return False, submitter_id, f"File write error: {str(e)}"

# ================= MAIN CONTROLLER =================

def process_patients():
    if not os.path.exists(INPUT_CSV_PATH):
        print(f"Error: Input file not found at {INPUT_CSV_PATH}")
        return

    df = pd.read_csv(INPUT_CSV_PATH)
    
    # 预处理：构建待处理列表
    all_patients = []
    for _, row in df.iterrows():
        raw_id = str(row.get('PID'))
        if pd.isna(raw_id) or str(raw_id).strip() == "Not Available":
            continue

        sid_str = str(raw_id).strip()
        
        all_patients.append({
            'PID': sid_str,
            'clinical': row.get('clinical'),
            'pathology': row.get('pathology'),
            'treatment': row.get('treatment'),
        })

    # 断点续传逻辑
    final_data = {}
    if os.path.exists(OUTPUT_JSON_PATH):
        try:
            with open(OUTPUT_JSON_PATH, 'r', encoding='utf-8') as f:
                content = f.read()
                if content.strip():
                    final_data = json.loads(content)
            print(f"Resuming with {len(final_data)} existing records.")
        except Exception as e:
            print(f"Error loading existing JSON: {e}. Starting fresh.")
            final_data = {}

    # 去重：过滤掉已经存在于 JSON 中的 ID
    patients_to_process = []
    seen_ids = set() # 防止 CSV 内部重复
    for p in all_patients:
        sid = p['PID']
        if sid not in final_data and sid not in seen_ids:
            patients_to_process.append(p)
            seen_ids.add(sid)

    print(f"Remaining unique patients to process: {len(patients_to_process)}")

    if not patients_to_process:
        print("All patients processed!")
        return

    # 多线程执行
    file_lock = threading.Lock()
    
    print(f"Starting execution with {MAX_WORKERS} threads...")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_id = {
            executor.submit(process_single_patient, p_data, final_data, file_lock): p_data['PID'] 
            for p_data in patients_to_process
        }

        for future in tqdm(as_completed(future_to_id), total=len(patients_to_process), desc="Patient Processing"):
            sid = future_to_id[future]
            try:
                success, _, msg = future.result()
                if not success:
                    with open(FAILED_LOG_PATH, "a") as err_f:
                        err_f.write(f"{sid}: {msg}\n")
            except Exception as e:
                print(f"Critical thread error for {sid}: {e}")

if __name__ == "__main__":
    process_patients()