import pandas as pd
import json
import os
import time
import threading
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ================= CONFIGURATION =================
API_KEY = os.environ.get("QWEN_API_KEY", "") 
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "qwen-turbo"

# File Paths (TCGA-LUAD)
INPUT_CSV_PATH = "/home/Guanjq/NewWork/MedAlignFusion/Data/TCGA-LUAD/processed/multimodal_texts.csv"
OUTPUT_JSON_PATH = "/home/Guanjq/NewWork/MedAlignFusion/Data/TCGA-LUAD/processed/medical_analysis_qwen.json"
FAILED_LOG_PATH = "failed_ids_luad_qwen.log"

# Threading Configuration
MAX_WORKERS = 10 

# ================= MAPPING & VALIDATION LOGIC =================

DATA_TYPE_MAPPING = {
    # Clinical
    'clinical data': 'clinical', 'clinical': 'clinical',
    # Treatment
    'treatment data': 'treatment', 'treatment': 'treatment',
    # Pathology
    'pathological data': 'pathology', 'pathological': 'pathology', 'pathology': 'pathology',
    # Genomics
    'genomic data': 'genomics', 'genomics data': 'genomics',
    'genomic': 'genomics', 'genomics': 'genomics',
    'molecular data': 'genomics', 'mgenomics': 'genomics'
}

VALID_KEYS = {'clinical', 'treatment', 'pathology', 'genomics'}

def map_data_type(data_string):
    if not isinstance(data_string, str): return None
    normalized = data_string.lower().strip()
    if normalized in DATA_TYPE_MAPPING: return DATA_TYPE_MAPPING[normalized]
    cleaned = normalized.replace("data", "").strip()
    if cleaned in DATA_TYPE_MAPPING: return DATA_TYPE_MAPPING[cleaned]
    if 'clinical' in normalized: return 'clinical'
    elif 'treatment' in normalized: return 'treatment'
    elif 'pathology' in normalized or 'pathological' in normalized: return 'pathology'
    elif 'genomic' in normalized or 'molecular' in normalized: return 'genomics'
    return None

def validate_and_normalize_response(parsed_json):
    """
    Validates that:
    1. All modalPairs map correctly to the 4 standard keys.
    2. There are exactly C(4,2) = 6 unique pairs.
    """
    if not isinstance(parsed_json, list): raise ValueError("Output must be a list")
    normalized_list = []
    seen_pairs = set()

    # parse json pairs
    # check relationships and check survival analysis
    for entry in parsed_json:
        if "modalPairs" not in entry or not isinstance(entry["modalPairs"], list):
            raise ValueError("Missing or invalid 'modalPairs' field")
        raw_pair = entry["modalPairs"]
        if len(raw_pair) != 2: raise ValueError(f"Pair does not contain exactly 2 elements: {raw_pair}")

        m1 = map_data_type(raw_pair[0])
        m2 = map_data_type(raw_pair[1])

        if m1 not in VALID_KEYS or m2 not in VALID_KEYS:
            raise ValueError(f"Could not map pair {raw_pair} to valid keys. Got: {m1}, {m2}")
        if m1 == m2: raise ValueError(f"Self-pair detected: {m1}-{m2}")

        sorted_pair = tuple(sorted([m1, m2]))
        if sorted_pair in seen_pairs: raise ValueError(f"Duplicate pair detected: {sorted_pair}")
        
        seen_pairs.add(sorted_pair)
        entry["modalPairs"] = [m1, m2]

        if "relationship" not in entry or "survival" not in entry:
            raise ValueError("Missing 'relationship' or 'survival' field")

        normalized_list.append(entry)

    # Final Count Check: C(4, 2) = 6
    if len(seen_pairs) != 6:
        raise ValueError(f"Expected exactly 6 unique pairs, found {len(seen_pairs)}. Pairs: {seen_pairs}")

    return normalized_list

# ================= CLIENT SETUP =================
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def construct_prompt(clinical, pathology, treatment, genomics):
    """
    Constructs the prompt string with the specific patient data.
    """    
    prompt = f"""
You are a professional physician with expertise in medical knowledge across various departments. 
This is data from a lung tumor patient in the TCGA-LUAD dataset to analyze the patient's survival risk.

The clinical data: {clinical}

The pathological data: {pathology}

The treatment data: {treatment}

The genomics data: {genomics}

You need to evaluate the degree of association between the two modalities of data for the patient's survival analysis, providing an integer score from 0 to 10, where 0 indicates the lowest association and 10 the highest. You need to generate for all pairs given to you:
1. Clinical - Pathology
2. Clinical - Treatment
3. Clinical - Genomics
4. Pathology - Treatment
5. Pathology - Genomics
6. Treatment - Genomics


As a professional physician, you must integrate both modalities of data to assess the patient's survival risk. 
Strictly use the following names for modalities in the "modalPairs" list: "clinical", "pathology", "treatment", "genomics", and your output format should be in json format:

```
[
    {{
        "modalPairs": ["Modal1", "Modal2"],
        "score": [An integer representing the association score between the two modalities of data, encouraged to be different among pairs],
        "relationship":  [A text paragraph analyzing the association between the two modalities of data, including your perspective on their relationship as detailed as possible],
        "survival": [A survival risk analysis integrating both modalities of data as detailed as possible],
    }},
    ... Output all modalpairs without overlap.
]
```
Stop after generating above output. Return ONLY the JSON.
"""
    
    return prompt

def clean_and_parse_json(response_text):
    try:
        text = response_text.strip()
        start_idx = text.find('[')
        end_idx = text.rfind(']')
        if start_idx == -1 or end_idx == -1: raise ValueError("No JSON list brackets [] found")
        return json.loads(text[start_idx : end_idx + 1])
    except Exception as e:
        raise e

# ================= WORKER FUNCTION =================

def process_single_patient(row_data, final_data_ref, file_lock):
    """
    Worker function to process a single patient.
    """
    submitter_id = row_data['submitter_id']
    
    # Extract Modalities
    clinical = row_data.get('clinical')
    pathology = row_data.get('pathology')
    treatment = row_data.get('treatment')
    genomics = row_data.get('genomics')

    # --- CRITICAL ASSERTION ---
    # 严格验证输入数据不为 None
    try:
        assert None not in [clinical, pathology, treatment, genomics], \
            f"Input Validation Failed: One or more modalities are None for {submitter_id}"
    except AssertionError as e:
        return False, submitter_id, str(e)
    # --------------------------

    prompt_text = construct_prompt(clinical, pathology, treatment, genomics)

    MAX_RETRIES = 10
    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {'role': 'system', 'content': 'You are a helpful medical AI assistant that outputs strict JSON with specific keys.'},
                    {'role': 'user', 'content': prompt_text}
                ],
                timeout=120
            )
            
            response_content = completion.choices[0].message.content
            parsed_raw = clean_and_parse_json(response_content)
            normalized_result = validate_and_normalize_response(parsed_raw)
            
            # --- CRITICAL SECTION: ATOMIC FILE WRITE ---
            # 使用临时文件 + 原子替换，确保文件不损坏
            with file_lock:
                final_data_ref[submitter_id] = normalized_result
                
                temp_file = OUTPUT_JSON_PATH + ".tmp"
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump(final_data_ref, f, ensure_ascii=False, indent=4)
                
                os.replace(temp_file, OUTPUT_JSON_PATH)
            # ------------------------------------

            return True, submitter_id, "Success"

        except Exception as e:
            last_error = str(e)
            time.sleep(1 + attempt) # 指数退避
    
    return False, submitter_id, last_error

# ================= MAIN CONTROLLER =================

def process_patients():
    if not os.path.exists(INPUT_CSV_PATH):
        print(f"Error: Input file not found at {INPUT_CSV_PATH}")
        return

    df = pd.read_csv(INPUT_CSV_PATH)
    df = df.fillna("Not Available")
    
    # 构建待处理列表，进行预处理
    all_patients = []
    for _, row in df.iterrows():
        # TCGA-LUAD 和 LUSC 通常使用 'cases.submitter_id'
        raw_id = row.get('cases.submitter_id')
        
        if pd.isna(raw_id) or raw_id == "Not Available":
            continue

        # 强制转换为字符串并去空格，这是防止重复跑数据的关键
        sid_str = str(raw_id).strip()
        
        all_patients.append({
            'submitter_id': sid_str,
            'clinical': row.get('clinical'),
            'pathology': row.get('pathology'),
            'treatment': row.get('treatment'),
            'genomics': row.get('genomics')
        })

    # Resume Logic & Sanitization
    final_data = {}
    if os.path.exists(OUTPUT_JSON_PATH):
        try:
            with open(OUTPUT_JSON_PATH, 'r', encoding='utf-8') as f:
                content = f.read()
                if content.strip():
                    loaded_data = json.loads(content)
                    print(f"Validating {len(loaded_data)} existing records...")
                    for sid, record in loaded_data.items():
                        try:
                            # 确保 Key 是干净的
                            clean_sid = str(sid).strip()
                            final_data[clean_sid] = validate_and_normalize_response(record)
                        except Exception as e:
                            print(f"  [Invalid Record] {sid}: {e} -> Will re-process.")
            
            # Sanitization Step: 立即重写文件以清洗潜在的重复键或格式问题
            if final_data:
                print(f"Sanitizing output file: re-saving {len(final_data)} clean, unique records...")
                with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
                    json.dump(final_data, f, ensure_ascii=False, indent=4)

        except Exception as e:
            print(f"Error loading existing JSON: {e}. Starting fresh.")
            final_data = {}

    print(f"Resuming with {len(final_data)} valid records.")

    # Deduplication Filter
    patients_to_process = []
    seen_in_batch = set()

    for p in all_patients:
        sid = p['submitter_id']
        
        # 1. 检查是否已处理
        if sid in final_data:
            continue
            
        # 2. 检查批次内重复 (CSV可能有重复行)
        if sid in seen_in_batch:
            continue
        
        patients_to_process.append(p)
        seen_in_batch.add(sid)

    print(f"Remaining unique patients to process: {len(patients_to_process)}")

    if not patients_to_process:
        print("All patients processed!")
        return

    # Multi-threading Execution
    file_lock = threading.Lock()
    
    print(f"Starting execution with {MAX_WORKERS} threads...")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_id = {
            executor.submit(process_single_patient, p_data, final_data, file_lock): p_data['submitter_id'] 
            for p_data in patients_to_process
        }

        for future in tqdm(as_completed(future_to_id), total=len(patients_to_process), desc="Concurrent Processing"):
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