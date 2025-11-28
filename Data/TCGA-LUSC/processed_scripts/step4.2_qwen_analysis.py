import pandas as pd
import json
import os
import time
from openai import OpenAI
from tqdm import tqdm

# ================= CONFIGURATION =================
API_KEY = os.environ.get("QWEN_API_KEY", None)
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "qwen-turbo"

# File Paths
INPUT_CSV_PATH = "/home/Guanjq/NewWork/MedAlignFusion/Data/TCGA-LUSC/processed/multimodal_texts.csv"
OUTPUT_JSON_PATH = "/home/Guanjq/NewWork/MedAlignFusion/Data/TCGA-LUSC/processed/medical_analysis_qwen.json"

# ================= MAPPING & VALIDATION LOGIC =================

DATA_TYPE_MAPPING = {
    # Clinical
    'clinical data': 'clinical',
    'clinical': 'clinical',
    # Treatment
    'treatment data': 'treatment',
    'treatment': 'treatment',
    # Pathology
    'pathological data': 'pathology',
    'pathological': 'pathology',
    'pathology': 'pathology',
    # Genomics
    'genomic data': 'genomics',
    'genomics data': 'genomics',
    'genomic': 'genomics',
    'genomics': 'genomics',
    'molecular data': 'genomics',
    'mgenomics': 'genomics'
}

VALID_KEYS = {'clinical', 'treatment', 'pathology', 'genomics'}

def map_data_type(data_string):
    """
    Standardizes the data type string to one of the 4 valid keys.
    Returns None if mapping fails.
    """
    if not isinstance(data_string, str):
        return None
        
    # Normalize: lowercase and strip whitespace
    normalized = data_string.lower().strip()
    
    # 1. Direct dictionary lookup
    if normalized in DATA_TYPE_MAPPING:
        return DATA_TYPE_MAPPING[normalized]
    
    # 2. Heuristic removal of "data" if not in dict
    cleaned = normalized.replace("data", "").strip()
    if cleaned in DATA_TYPE_MAPPING:
        return DATA_TYPE_MAPPING[cleaned]

    # 3. Keyword matching (Safety net)
    if 'clinical' in normalized:
        return 'clinical'
    elif 'treatment' in normalized:
        return 'treatment'
    elif 'pathology' in normalized or 'pathological' in normalized:
        return 'pathology'
    elif 'genomic' in normalized or 'molecular' in normalized:
        return 'genomics'
        
    return None

def validate_and_normalize_response(parsed_json):
    """
    Validates that:
    1. All modalPairs map correctly to the 4 standard keys.
    2. There are exactly C(4,2) = 6 unique pairs.
    3. No self-loops (e.g. clinical-clinical).
    
    Returns the normalized list if valid, otherwise raises ValueError.
    """
    if not isinstance(parsed_json, list):
        raise ValueError("Output must be a list")

    normalized_list = []
    seen_pairs = set()

    for entry in parsed_json:
        if "modalPairs" not in entry or not isinstance(entry["modalPairs"], list):
            raise ValueError("Missing or invalid 'modalPairs' field")
        
        raw_pair = entry["modalPairs"]
        if len(raw_pair) != 2:
            raise ValueError(f"Pair does not contain exactly 2 elements: {raw_pair}")

        # Map keys
        m1 = map_data_type(raw_pair[0])
        m2 = map_data_type(raw_pair[1])

        # Check validity
        if m1 not in VALID_KEYS or m2 not in VALID_KEYS:
            raise ValueError(f"Could not map pair {raw_pair} to valid keys. Got: {m1}, {m2}")
        
        if m1 == m2:
            raise ValueError(f"Self-pair detected: {m1}-{m2}")

        # Check Uniqueness (sort to handle [A,B] same as [B,A])
        sorted_pair = tuple(sorted([m1, m2]))
        
        if sorted_pair in seen_pairs:
            # We treat duplicates as an error because we want exactly 6 unique pairs covering the combination
            # Alternatively, you could skip, but strictly we need 6 unique distinct pairs.
            raise ValueError(f"Duplicate pair detected: {sorted_pair}")
        
        seen_pairs.add(sorted_pair)
        
        # Update entry with normalized keys
        entry["modalPairs"] = [m1, m2]
        normalized_list.append(entry)

    # Final Count Check: C(4, 2) = 6
    if len(seen_pairs) != 6:
        raise ValueError(f"Expected exactly 6 unique pairs, found {len(seen_pairs)}. Pairs found: {seen_pairs}")

    return normalized_list

# ================= LLM CLIENT & LOGIC =================

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
)


def construct_prompt(clinical, pathology, treatment, genomics):
    """
    Constructs the prompt string with the specific patient data.
    """
    num_pairs = 6
    
    prompt = f"""
You are a professional physician with expertise in medical knowledge across various departments. 
This is data from a Lung Squamous Cell Carcinoma tumor patient in the TCGA-LUSC dataset to analyze the patient's survival risk.

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
    """
    Cleans the LLM response by extracting the substring between the first '[' and last ']',
    ignoring any text before or after the JSON array.
    """
    try:
        text = response_text.strip()
        
        # Robust extraction: find the outer brackets
        start_idx = text.find('[')
        end_idx = text.rfind(']')
        
        if start_idx == -1 or end_idx == -1:
            raise ValueError("No JSON list brackets [] found in response")
            
        json_str = text[start_idx : end_idx + 1]
        
        return json.loads(json_str)
    except Exception as e:
        raise e

def process_patients():
    # 1. Load Data
    if not os.path.exists(INPUT_CSV_PATH):
        print(f"Error: Input file not found at {INPUT_CSV_PATH}")
        return

    df = pd.read_csv(INPUT_CSV_PATH)
    df = df.fillna("Not Available")

    # 2. Load existing results (Resume capability) with Strict Validation
    final_data = {}
    if os.path.exists(OUTPUT_JSON_PATH):
        try:
            with open(OUTPUT_JSON_PATH, 'r', encoding='utf-8') as f:
                content = f.read()
                if content.strip():
                    loaded_data = json.loads(content)
                    print(f"Checking {len(loaded_data)} records from disk for validity...")
                    
                    invalid_count = 0
                    for sid, record in loaded_data.items():
                        try:
                            # Validate immediately upon loading.
                            # If this fails, it raises ValueError and we go to except block.
                            # We save the normalized version to ensure consistency.
                            final_data[sid] = validate_and_normalize_response(record)
                        except ValueError as e:
                            # If data on disk is invalid, we do NOT add it to final_data.
                            # This ensures it will be re-processed in the loop below.
                            print(f"  [Invalid Record Found] ID {sid}: {e} -> Scheduled for re-generation.")
                            invalid_count += 1
                            
                    print(f"Resuming with {len(final_data)} valid records. (Found {invalid_count} invalid records to re-run)")
                    
        except json.JSONDecodeError:
            print("Output file exists but is empty or corrupt. Starting fresh.")
            final_data = {}

    # 3. Iterate through patients
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing Patients"):
        submitter_id = row.get('cases.submitter_id')
        
        # Check if already processed (only valid data is in final_data now)
        if submitter_id in final_data:
            continue

        # Extract Modalities
        clinical = row.get('clinical', None)
        pathology = row.get('pathology', None)
        treatment = row.get('treatment', None)
        genomics = row.get('genomics', None)

        assert None not in [clinical, pathology, treatment, genomics], "One or more modalities are missing"

        prompt_text = construct_prompt(clinical, pathology, treatment, genomics)

        # Retry Loop
        MAX_RETRIES = 10
        success = False
        last_error = None

        for attempt in range(MAX_RETRIES):
            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {'role': 'system', 'content': 'You are a helpful medical AI assistant that outputs strict JSON with specific keys.'},
                        {'role': 'user', 'content': prompt_text}
                    ],
                )
                
                response_content = completion.choices[0].message.content
                
                # 1. Parse JSON
                parsed_raw = clean_and_parse_json(response_content)
                
                # 2. Validate and Normalize (Key mapping & C_4^2 check)
                normalized_result = validate_and_normalize_response(parsed_raw)
                
                # If we get here, result is valid
                final_data[submitter_id] = normalized_result
                
                # Save to disk immediately
                with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
                    json.dump(final_data, f, ensure_ascii=False, indent=4)
                
                success = True
                break # Exit retry loop
            
            except Exception as e:
                last_error = str(e)
                # Optional: print simple log for debugging retries
                # print(f"  [Retry {attempt+1}/{MAX_RETRIES}] ID {submitter_id}: {e}")
                time.sleep(1) # Short pause before retry

        # If failed after all retries
        if not success:
            error_msg = f"FAILED {submitter_id} after {MAX_RETRIES} attempts. Reason: {last_error}\n"
            print(error_msg.strip())
            # Optionally write failed IDs to a log file
            with open("failed_ids.log", "a") as err_f:
                err_f.write(f"{submitter_id}: {last_error}\n")

if __name__ == "__main__":
    process_patients()