import pandas as pd
import json
import os
import time
from openai import OpenAI
from tqdm import tqdm

# ================= CONFIGURATION =================
API_KEY = os.environ.get("QWEN_API_KEY", "sk-PLACEHOLDER_IF_ENV_NOT_SET") 
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "qwen-turbo"

# File Paths
INPUT_CSV_PATH = "/home/Guanjq/NewWork/MedAlignFusion/Data/TCGA-LUAD/processed/multimodal_texts.csv"
OUTPUT_JSON_PATH = "/home/Guanjq/NewWork/MedAlignFusion/Data/TCGA-LUAD/processed/qwen_analysis.json"

# ================= SETUP CLIENT =================
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

As a professional physician, you must integrate both modalities of data to assess the patient's survival risk. Your output format should be in json format:

```
[
    {{
        "modalPairs": ["Modal1", "Modal2"],
        "score": [An integer representing the association score between the two modalities of data],
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
        # Pass the exception up so we can log the specific reason
        raise e

def process_patients():
    # 1. Load Data
    if not os.path.exists(INPUT_CSV_PATH):
        print(f"Error: Input file not found at {INPUT_CSV_PATH}")
        return

    df = pd.read_csv(INPUT_CSV_PATH)
    df = df.fillna("Not Available")

    # 2. Load existing results (Resume capability)
    final_data = {}
    if os.path.exists(OUTPUT_JSON_PATH):
        try:
            with open(OUTPUT_JSON_PATH, 'r', encoding='utf-8') as f:
                final_data = json.load(f)
            print(f"Loaded {len(final_data)} existing records. Resuming...")
        except json.JSONDecodeError:
            print("Output file exists but is empty or corrupt. Starting fresh.")
            final_data = {}

    # 3. Iterate through patients
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing Patients"):
        submitter_id = row.get('cases.submitter_id')
        
        # Skip if already processed
        if submitter_id in final_data:
            continue

        # Extract Modalities
        clinical = row.get('Clinical', 'N/A')
        pathology = row.get('Pathology', 'N/A')
        treatment = row.get('Treatment', 'N/A')
        genomics = row.get('Genomics', 'N/A')

        prompt_text = construct_prompt(clinical, pathology, treatment, genomics)

        # Retry Loop
        max_retries = 3
        success = False
        last_error = None

        for attempt in range(max_retries):
            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {'role': 'system', 'content': 'You are a helpful medical AI assistant that outputs strict JSON.'},
                        {'role': 'user', 'content': prompt_text}
                    ],
                    temperature=0.2,
                )
                
                response_content = completion.choices[0].message.content
                parsed_result = clean_and_parse_json(response_content)
                
                if parsed_result:
                    # Save to memory
                    final_data[submitter_id] = parsed_result
                    
                    # Save to disk
                    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
                        json.dump(final_data, f, ensure_ascii=False, indent=4)
                    
                    success = True
                    break # Exit retry loop
            
            except Exception as e:
                last_error = str(e)
                # print(f"  Attempt {attempt + 1}/{max_retries} failed for {submitter_id}: {e}")
                time.sleep(1) # Short pause before retry

        # If failed after all retries
        if not success:
            error_msg = f"FAILED {submitter_id} after {max_retries} attempts. Reason: {last_error}\n"
            print(error_msg.strip())

if __name__ == "__main__":
    process_patients()