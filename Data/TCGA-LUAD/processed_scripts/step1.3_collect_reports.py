import os
import re # 导入正则表达式模块
import pandas as pd
import pdfplumber
from tqdm import tqdm # 用于显示漂亮的进度条

# --- 用户配置 ---
# 请将此路径修改为您 TCGA-LUAD 报告的根目录
# 也就是包含所有患者 ID 文件夹的 'reports' 目录
BASE_REPORTS_DIR = "/home/Guanjq/NewWork/MedAlignFusion/Data/TCGA-LUAD/source/reports"
OUTPUT_CSV = "/home/Guanjq/NewWork/MedAlignFusion/Data/TCGA-LUAD/processed/tcga_luad_reports.csv"
# --- 用户配置结束 ---

def extract_pdf_text(filepath):
    """
    从单个 PDF 文件中提取所有文本，并清理符号。
    """
    if filepath is None or not os.path.exists(filepath):
        return None
        
    full_text = ""
    try:
        with pdfplumber.open(filepath) as pdf:
            # 遍历 PDF 的每一页
            for page in pdf.pages:
                # 提取当前页的文本
                text = page.extract_text()
                if text:
                    full_text += text + "\n"
        
        # --- 新增的清理步骤 ---
        if full_text:
            # 1. 替换连续两个或更多非字母、非数字、非空白的符号为空格
            #    [^\w\s] 匹配任何不是（^）字母数字（\w）或空白（\s）的字符
            #    {2,}    匹配 2 次或更多
            cleaned_text = re.sub(r'([^\w\s]){2,}', ' ', full_text)
            
            # 2. 额外清理：将多个空白字符（包括换行符）替换为单个空格
            cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
            
            # 3. 移除开头和结尾的空白
            return cleaned_text.strip()
        else:
            return None # 如果 full_text 为空，返回 None
        # --- 清理步骤结束 ---

    except Exception as e:
        print(f"  [警告] 无法读取 PDF {filepath}: {e}")
        return None

def read_annotation_notes_from_txt(filepath):
    """
    读取 .txt 文件 (TSV 格式), 并提取 'notes' 列的内容。
    """
    if filepath is None or not os.path.exists(filepath):
        return None
        
    try:
        # 1. 使用 pandas 读取 TSV (Tab 分隔)
        # on_bad_lines='skip' 会跳过任何格式错误的行
        df = pd.read_csv(filepath, sep='\t', on_bad_lines='skip')
        
        # 2. 检查 'notes' 列是否存在
        if 'notes' in df.columns:
            # 3. 提取 'notes' 列, 过滤掉
            #    可能的空值 (NaT/None), 转换为字符串
            notes_list = df['notes'].dropna().astype(str).tolist()
            
            # 4. 将所有 notes 合并成一个字符串, 用换行符分隔
            if notes_list:
                return "\n".join(notes_list)
            else:
                return None # 'notes' 列存在, 但没有内容
        else:
            print(f"  [警告] TXT 文件 {filepath} 中未找到 'notes' 列。")
            return None # 文件存在, 但没有 'notes' 列

    except pd.errors.EmptyDataError:
         print(f"  [警告] TXT 文件 {filepath} 为空。")
         return None
    except Exception as e:
        print(f"  [警告] 无法解析 TXT {filepath}: {e}")
        return None

def create_reports_dataframe(base_dir):
    """
    遍历基础目录，提取信息并构建 DataFrame。
    """
    if not os.path.exists(base_dir):
        print(f"错误：目录 '{base_dir}' 不存在。")
        print("请检查 'BASE_REPORTS_DIR' 变量是否设置正确。")
        return pd.DataFrame() # 返回一个空 DataFrame

    data_rows = []
    
    # 获取所有子目录（即 patient_id 文件夹）
    patient_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    if not patient_dirs:
        print(f"错误：在 '{base_dir}' 中未找到任何患者ID子目录。")
        return pd.DataFrame()

    print(f"在 '{base_dir}' 中找到了 {len(patient_dirs)} 个患者ID目录。")
    print("开始处理文件...")

    # 使用 tqdm 创建一个进度条
    for dir_name in tqdm(patient_dirs, desc="处理患者数据"):
        patient_dir_path = os.path.join(base_dir, dir_name)
        
        pdf_path = None
        txt_path = None
        patient_id = None
        
        # 扫描文件夹内的文件
        for filename in os.listdir(patient_dir_path):
            if filename.lower().endswith('.pdf'):
                pdf_path = os.path.join(patient_dir_path, filename)
                patient_id = filename.split(".")[0]
                
            elif filename == 'annotations.txt':
                txt_path = os.path.join(patient_dir_path, filename)
        
        if patient_id is None:
            continue

        # 提取数据
        report_text = extract_pdf_text(pdf_path)
        # 更新了此处的函数调用
        annotation_text = read_annotation_notes_from_txt(txt_path)
        
        # 添加到列表
        row = {
            'patient_id': patient_id,
            'report_text': report_text, # PDF 中的文本 (已清理)
            'annotation_text': annotation_text # annotations.txt 中的文本 (现在只有 notes)
        }
        data_rows.append(row)

    # 创建 DataFrame
    df = pd.DataFrame(data_rows)
    return df

if __name__ == "__main__":
    print("--- TCGA 报告数据处理开始 ---")
    
    df = create_reports_dataframe(BASE_REPORTS_DIR)
    
    if not df.empty:
        print(f"\n处理完毕。共处理 {len(df)} 条患者记录。")
        
        # 打印 DataFrame 的基本信息
        print("\nDataFrame 概览 (前5行):")
        print(df.head())
        
        print("\nDataFrame 信息:")
        df.info()
        
        # 检查有多少 PDF 和 TXT 被成功读取
        print(f"\n成功读取的 PDF 报告数量: {df['report_text'].notna().sum()}")
        print(f"成功读取的 TXT 注释数量: {df['annotation_text'].notna().sum()}")
        
        # 保存到 CSV
        try:
            # 使用 utf-8-sig 编码确保 Excel 打开 CSV 时中文无乱码
            df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
            print(f"\n数据已成功保存到: {OUTPUT_CSV}")
        except Exception as e:
            print(f"\n保存 CSV 文件时出错: {e}")
            
    else:
        print("未处理任何数据，请检查您的 'BASE_REPORTS_DIR' 配置。")