import os
import re
import pandas as pd
import pdfplumber
from tqdm import tqdm
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import pypdf   

# --- 用户配置 ---
# 路径配置
BASE_REPORTS_DIR = "/home/Guanjq/NewWork/MedAlignFusion/Data/TCGA-LUAD/source/reports"
OUTPUT_CSV = "/home/Guanjq/NewWork/MedAlignFusion/Data/TCGA-LUAD/processed/tcga_luad_reports.csv"

# API 配置 (Qwen/通义千问)
API_KEY = os.environ.get("QWEN_API_KEY", None)
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "qwen-turbo"

# 多线程配置
MAX_WORKERS = 10 
# --- 用户配置结束 ---

# 初始化 OpenAI 客户端
client = None
if API_KEY and "sk-" in API_KEY:
    client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
    )
else:
    print("[警告] 未检测到有效的 API Key。")

logging.getLogger("pdfminer").setLevel(logging.ERROR)

def extract_pdf_text(filepath):
    """
    健壮的 PDF 文本提取函数。
    策略：优先使用 pdfplumber，如果因格式错误(如 invalid float value)失败，
    则降级使用 pypdf 进行提取。
    """
    if filepath is None or not os.path.exists(filepath):
        return None
    
    text_content = ""
    
    # --- 方法 A: 尝试 pdfplumber ---
    try:
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                # 某些页面可能解析失败，单独捕获页面的错误
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text_content += page_text + "\n"
                except Exception:
                    continue  # 单页失败不影响整体
    except Exception as e:
        # pdfplumber 彻底失败（比如文件头损坏），记录一下但不打印巨量日志
        pass

    # --- 方法 B: 如果 pdfplumber 没提取到内容，使用 pypdf 救急 ---
    # pypdf 对 '/P0' 这种颜色错误容忍度极高
    if not text_content.strip():
        try:
            reader = pypdf.PdfReader(filepath)
            temp_text = []
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    temp_text.append(extracted)
            text_content = "\n".join(temp_text)
        except Exception as e:
            # 如果两个库都挂了，那这个 PDF 可能是损坏的二进制文件
            print(f"[提取失败] {os.path.basename(filepath)} 无法被解析。")
            return None

    # --- 统一的后处理 ---
    if text_content:
        # 清洗特殊符号和多余空格
        cleaned_text = re.sub(r'([^\w\s]){2,}', ' ', text_content)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
        return cleaned_text.strip()
    
    return None

def read_annotation_notes_from_txt(filepath):
    """读取 .txt 文件 (TSV 格式), 并提取 'notes' 列的内容。"""
    if filepath is None or not os.path.exists(filepath):
        return None
    try:
        df = pd.read_csv(filepath, sep='\t', on_bad_lines='skip')
        if 'notes' in df.columns:
            notes_list = df['notes'].dropna().astype(str).tolist()
            return "\n".join(notes_list) if notes_list else None
        return None
    except:
        return None

def polish_report_with_llm(report_text, annotation_text):
    """调用 Qwen API 对报告进行润色。"""
    if not client:
        return None
    
    r_text = report_text if report_text else 'N/A'
    a_text = annotation_text if annotation_text else 'N/A'
    
    input_content = f"[Raw PDF Report]:\n{r_text}\n\n[Annotations]:\n{a_text}"

    system_prompt = """
You are a senior pathologist and oncology expert. 
You are processing a raw pathology report containing OCR noise and related annotation information.
Your task is to reconstruct and polish this information to generate a structured, clear, and professional pathology review report.

Please follow these steps:
1. **Cleaning and Filtering**: Remove OCR garbled text, headers, footers, administrative information irrelevant to the specific medical condition, or formatting characters.
2. **Information Integration**: Logically integrate the content of the PDF report with the Annotation notes.
3. **Knowledge Injection (Key)**:
   - Identify key pathological features in the report (e.g., Grade, Stage, Receptor Status, Histological Subtype, etc.).
   - When describing these features, briefly supplement their clinical significance in **Tumor Survival Analysis** (e.g., high expression of a certain marker is usually associated with poor prognosis).
   - **Important**: If the input report and annotations are empty or "N/A", please generate a **general educational summary** of key pathological factors in Breast Cancer (BRCA) that affect patient survival (e.g., ER/PR/HER2 status, Ki-67, Nuclear Grade), explaining their clinical relevance.
4. **Output Control**
   - The output format must be enclosed within <output> tags.
   - Maintain professional and objective language.
   - Do not output any pleasantries; output the report content directly.

Example Output Structure:
<output>
**Comprehensive Pathology Diagnostic Report**
[Organized Diagnostic Conclusion / General Knowledge Summary in Detailed]
[Detailed Pathological Description / Factor Explanations in Detailed]
[Survival Analysis Interpretation: Explain the potential impact of extracted key indicators on prognosis with comprehensive medical knowledge]
</output>

Stop after the output is complete.
"""

    max_retries = 10
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': input_content}
                ],
            )
            content = completion.choices[0].message.content
            match = re.search(r'<output>(.*?)</output>', content, re.DOTALL)
            if match:
                return match.group(1).strip()
        except Exception:
            continue
    return None

def get_real_patient_id_from_dir(dir_path):
    """
    快速扫描目录下的 PDF 文件名，提取真正的 TCGA ID。
    例如从 'TCGA-AC-A2FO.5A7EB73F....PDF' 中提取 'TCGA-AC-A2FO'
    """
    try:
        for f in os.listdir(dir_path):
            if f.lower().endswith('.pdf'):
                # 假设文件名以点号分割，第一部分是 ID
                return f.split('.')[0]
    except:
        return None
    return None

def process_single_patient(dir_name, base_dir):
    """单个患者处理逻辑"""
    patient_dir_path = os.path.join(base_dir, dir_name)
    pdf_path = None
    txt_path = None
    
    # 获取真正的 ID（文件名上的 ID）
    real_patient_id = get_real_patient_id_from_dir(patient_dir_path)
    
    # 如果找不到 PDF 里的 ID，只能降级使用文件夹名（通常不会发生）
    if not real_patient_id:
        real_patient_id = dir_name

    try:
        if not os.path.exists(patient_dir_path):
            return None

        for filename in os.listdir(patient_dir_path):
            if filename.lower().endswith('.pdf'):
                pdf_path = os.path.join(patient_dir_path, filename)
            elif filename == 'annotations.txt':
                txt_path = os.path.join(patient_dir_path, filename)
        
        report_text = extract_pdf_text(pdf_path)
        annotation_text = read_annotation_notes_from_txt(txt_path)
        
        polished_report = polish_report_with_llm(report_text, annotation_text)

        return {
            'patient_id': real_patient_id, # 确保这里返回的是 TCGA-XXX
            'original_report_text': report_text,
            'original_annotation_text': annotation_text,
            'llm_polished_report': polished_report
        }
    except Exception as e:
        print(f"Error processing {dir_name}: {e}")
        return None

def main():
    print("--- TCGA-BRCA 报告处理 (精确对齐修复版) ---")
    
    # 1. 建立“已完成”白名单 (Set)
    # 我们只关心 CSV 里的 patient_id (如 TCGA-C8-A12N)
    completed_ids = set()
    existing_data_map = {} # 用于最后合并，防止丢数据

    if os.path.exists(OUTPUT_CSV):
        try:
            print(f"读取 CSV: {OUTPUT_CSV}")
            df_existing = pd.read_csv(OUTPUT_CSV)
            
            # 遍历每一行，检查 LLM 是否有效
            for idx, row in df_existing.iterrows():
                pid = str(row['patient_id']).strip()
                llm_content = row.get('llm_polished_report')
                
                # 保存所有旧数据
                existing_data_map[pid] = row.to_dict()

                # 判断是否完成：内容非空且长度足够
                if pd.notna(llm_content) and isinstance(llm_content, str) and len(llm_content) > 10:
                    completed_ids.add(pid)
            
            print(f"CSV 中发现 {len(df_existing)} 条记录。")
            print(f"其中 {len(completed_ids)} 条已包含有效 LLM 润色内容 (将被跳过)。")
            
        except Exception as e:
            print(f"读取 CSV 出错: {e}")
            return
    else:
        print("未找到 CSV，全量运行。")

    # 2. 预扫描目录，构建任务列表
    # 关键步骤：必须打开文件夹看一眼 PDF 文件名，才能知道它对应的 ID 是什么
    all_dirs = [d for d in os.listdir(BASE_REPORTS_DIR) if os.path.isdir(os.path.join(BASE_REPORTS_DIR, d))]
    tasks_to_process = []
    
    print(f"\n正在预扫描 {len(all_dirs)} 个目录以匹配 ID (Pre-scanning)...")
    
    # 这一步是单线程的，但只是读文件名，速度很快
    for dir_name in tqdm(all_dirs, desc="ID匹配中"):
        full_dir_path = os.path.join(BASE_REPORTS_DIR, dir_name)
        
        # 从文件夹内的 PDF 获取真正的 TCGA ID
        real_id = get_real_patient_id_from_dir(full_dir_path)
        
        if not real_id:
            # 如果没有 PDF 或者提取失败，这可能是一个坏数据，或者我们还是把它加进去尝试处理一下
            # print(f"警告: 目录 {dir_name} 下未找到 PDF 或 ID 解析失败")
            tasks_to_process.append(dir_name)
            continue
            
        # 核心判断：如果这个真正的 ID 已经在 completed_ids 里，就跳过
        if real_id in completed_ids:
            continue
        else:
            # 没在白名单里，说明是漏网之鱼，或者是空的
            tasks_to_process.append(dir_name)

    todo_count = len(tasks_to_process)
    skipped_count = len(all_dirs) - todo_count
    
    print(f"\n--- 任务统计 ---")
    print(f"总目录数: {len(all_dirs)}")
    print(f"已完成 (跳过): {skipped_count}")
    print(f"待处理 (运行): {todo_count}")
    
    if todo_count == 0:
        print("所有文件均已处理完毕！")
        return

    # 3. 多线程处理
    print(f"\n启动线程池处理 {todo_count} 个任务...")
    new_results = []
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_dir = {executor.submit(process_single_patient, d, BASE_REPORTS_DIR): d for d in tasks_to_process}
        
        for future in tqdm(as_completed(future_to_dir), total=todo_count, desc="LLM处理中"):
            res = future.result()
            if res:
                # 无论成功失败，都记录下来
                new_results.append(res)
                if not res.get('llm_polished_report'):
                    tqdm.write(f"警告: {res['patient_id']} 处理后 LLM 仍为空")

    # 4. 合并数据
    print("\n正在合并并保存...")
    
    # 策略：以 existing_data_map 为基础，用 new_results 更新它
    for item in new_results:
        pid = item['patient_id']
        # 覆盖更新
        existing_data_map[pid] = item
        
    final_rows = list(existing_data_map.values())
    df_final = pd.DataFrame(final_rows)
    
    # 5. 保存
    if os.path.exists(OUTPUT_CSV):
        import shutil
        shutil.copy(OUTPUT_CSV, OUTPUT_CSV + ".bak")
        
    df_final.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    
    final_valid = df_final['llm_polished_report'].notna().sum()
    print(f"完成。当前 CSV 总行数: {len(df_final)}, 有效 LLM 记录: {final_valid}")

if __name__ == "__main__":
    main()