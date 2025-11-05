import os
import json
from typing import Dict, Any, List, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import random
import copy
import joblib # 导入 joblib

# 尝试导入 torch_geometric.data.Data
# 这样在 unpickle 'luad_data.pkl' 时不会出错
try:
    from torch_geometric.data import Data
except ImportError:
    print("警告: 无法导入 'torch_geometric.data.Data'。")
    print("请确保已安装 torch_geometric。")
    # 定义一个占位符，以防万一
    class Data:
        pass


class TCGA_LUAD_Dataset(Dataset):

    def _read_pickle(self, path: str) -> Any:
        """
        辅助函数，用于加载 pickle/joblib 文件。
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Pickle file not found at: {path}")
        
        print(f"Loading data file from: {path}")
        try:
            # --- 使用 joblib.load ---
            data = joblib.load(path)
            # ---------------------------
            return data
        except Exception as e:
            print(f"Error loading data file {path}: {e}")
            # --- 在 __init__ 中重新引发错误 ---
            raise

    def __init__(self, mode: str = "train", modalities: str = "all", fold: int = None):
        """
        Initializes the dataset.

        Args:
            mode (str): The dataset mode, one of 'train', 'valid', or 'test'.
            modalities (str): A comma-separated string of modalities to load,
                                e.g., "image,strong_related_text". "all" loads all available.
                                Supported: 'image', 'strong_related_text', 'weak_related_text'.
        """
        super().__init__()
        assert mode in ["train", "valid", "test"], "mode must be one of 'train', 'valid', or 'test'"

        random.seed(42)
        assert fold is not None, f"Fold ID must be specified."
        self.mode = mode
        """
        Data(x_img=[633, 1024], x_rna=[5, 1024], x_cli=[11, 1024],  data_type=[3], edge_index_image=[2, 4606], edge_index_rna=[2, 20], edge_index_cli=[2, 110])
        如何访问 (示例): data.x_img (访问图像特征)
                          data.edge_index_rna (访问RNA边)
                          data.sur_type (访问生存类型)
        """
        # !!! 注意：这里的路径是基于您提供的示例的相对路径
        # !!! 您可能需要将其修改为绝对路径或正确的相对路径
        self.dataset_dir = os.path.join(os.getcwd(), "../Data/TCGA-LUAD") 
        print(f"Warning: Using dataset directory: {self.dataset_dir}")
        print("Please ensure this path is correct or modify as needed.")

        self.data_pickle = self._read_pickle(os.path.join(self.dataset_dir, "source", "luad_data.pkl")) # DICT ID:torch_geometric.data.data.Data, 
        self.label_pickle = self._read_pickle(os.path.join(self.dataset_dir, "source", "luad_sur_and_time.pkl"))  # DICT ID: [death if 1 else 0, last time float number but means month]
        
        # 加载 5 折交叉验证数据
        # [fold] 会选择是第 0, 1, 2, 3, 还是 4 折
        split_pickle = self._read_pickle(os.path.join(self.dataset_dir, "source", "luad_split.pkl"))[fold]

        # Target Patient List
        # 根据之前的分析: 0=train, 1=valid, 2=test
        split_id = 0 if mode == 'train' else (1 if mode == 'valid' else 2)
        self.patient_ids = split_pickle[split_id].tolist()

        # --- Parse modalities ---
        self.modalities = self._parse_modalities(modalities)
        print(f"Dataset will be initialized for modalities: {self.modalities}")

        # --- Survival Analysis Parameters ---
        # --- 生存时间单位是 *月* ---
        self.five_years_in_months = 5 * 12.0  # 60.0 个月
        self.num_time_bins = 10
        # self.time_bins 结果: [ 0., 6., 12., 18., 24., 30., 36., 42., 48., 54., 60.]
        # 这匹配了 "半年一个分期bin"
        self.time_bins = np.linspace(0, self.five_years_in_months, self.num_time_bins + 1)
        print(f"Survival time bins (in Months): {self.time_bins}")
        # --- BUG 修复结束 ---

        # --- Load and preprocess all data sources ---
        self._load_data()
        print(f"Dataset for mode '{self.mode}' initialized. Found {len(self.patient_ids)} patients.")

    
    def _get_labels(self):
        """
        Returns a list of all labels (time bins) in the dataset.
        This is used by the SurvivalBalancedBatchSampler.
        (重写以匹配 __getitem__ 并使用 self.label_pickle)
        """
        labels_y = []
        for patient_id in self.patient_ids:
            try:
                # 1. 从 'luad_sur_and_time.pkl' 获取标签
                survival_info = self.label_pickle[patient_id]
                time_months = float(survival_info[1]) # 生存时间（月）

                # 2. 确保事件时间在 5 年 (60 个月) 内
                event_time = min(time_months, self.five_years_in_months)
                
                # 3. 将连续时间离散化到 10 个 bins (0-9)
                #    Y is the discrete time interval index
                time_bin = np.digitize(event_time, self.time_bins) - 1
                
                # 4. 确保索引在 [0, num_bins-1]
                time_bin = max(0, min(time_bin, self.num_time_bins - 1))
                
                labels_y.append(int(time_bin))

            except KeyError:
                # 如果一个 patient_id 存在于 split_pickle 但不存在于 label_pickle
                # 我们必须附加一个默认 bin，以保持与 self.patient_ids 的长度一致
                print(f"Warning: Patient ID {patient_id} not found in label_pickle. Assigning time_bin = 0 (default).")
                labels_y.append(0) # 附加一个默认 bin (0)
            except Exception as e:
                # 捕获其他潜在错误 (e.g., time_months 不是数字)
                print(f"Error processing label for {patient_id}: {e}. Assigning time_bin = 0 (default).")
                labels_y.append(0) # 附加一个默认 bin (0)
                
        return labels_y

    def _parse_modalities(self, modalities_str: str) -> List[str]:
        """Parses the modalities string into a list of valid modality keys."""

        valid_modalities = {
            "image-pathology", 
            "text-pathology", 
            "tabular-pathology-37", 
            "tabular-clinical-56", 
            "tabular-genomics-27",
            "genomics-genomics",
        }

        if modalities_str == "all":
            # 返回所有有效的模态
            return sorted(list(valid_modalities))
        
        # 解析逗号分隔的字符串
        requested_modalities = modalities_str.split(',')
        parsed_list = []
        for mod in requested_modalities:
            mod = mod.strip() # 清理空格
            if mod in valid_modalities:
                parsed_list.append(mod)
            else:
                print(f"Warning: Modality '{mod}' not recognized and will be skipped.")
        
        return parsed_list

    def _load_data(self):
        # 定义 processed 目录的路径
        processed_dir = os.path.join(self.dataset_dir, "processed")
        print(f"Loading data from processed directory: {processed_dir}")

        # 定义文件路径
        clinical_path = os.path.join(processed_dir, "clinical_data_aggregated.csv")
        genomics_path = os.path.join(processed_dir, "genomics_data_aggregated.csv")
        histology_path = os.path.join(processed_dir, "histology_data_aggregated.csv")
        reports_path = os.path.join(processed_dir, "tcga_luad_reports.csv")
        # time_info_path = os.path.join(processed_dir, "time_info_table.csv") # <-- 不再需要
        
        try:
            # --- 更新：加载时明确指定 dtype=str，以防止 pandas 自动转换 ---
            # 这样可以确保 '1.0' 和 '1' 都被视为字符串，直到 _process_row 处理
            self.clinical_df = pd.read_csv(clinical_path, dtype=str)
            print(f"Loaded clinical data with shape: {self.clinical_df.shape}")
            self.genomics_df = pd.read_csv(genomics_path, dtype=str)
            print(f"Loaded genomics data with shape: {self.genomics_df.shape}")
            self.histology_df = pd.read_csv(histology_path, dtype=str)
            print(f"Loaded histology data with shape: {self.histology_df.shape}")
            self.reports_df = pd.read_csv(reports_path) # 报告文本没有这个问题
            print(f"Loaded reports data with shape: {self.reports_df.shape}")
            # -----------------------------------------------------------
        except FileNotFoundError as e:
            print(f"!!! 错误: 文件未找到 !!!")
            print(f"加载 CSV 文件时出错: {e}")
            print(f"请确保文件存在于: {processed_dir}")
            raise
        except Exception as e:
            print(f"!!! 错误: 加载 CSV 文件时发生未知错误 !!!")
            print(f"错误详情: {e}")
            raise

        case_id_to_submitter = {}
        self.clinical_tabular_dict = {}  # patient key -> tabular
        
        # --- 更新后的辅助函数 ---
        # 定义一个辅助函数来处理行
        # 将所有非数字值 (NaN, 空值, 文本) 转换为 -1.0
        def _process_row(row, columns):
            tabular_data = []
            for col in columns:
                value = row[col]
                # 尝试将值转换为数字。
                # 'coerce' 会将所有无效值 (如 "", None, "NA", NaN) 转换为 NaT/NaN
                numeric_val = pd.to_numeric(value, errors='coerce')
                
                # 检查结果是否为 NaN
                if pd.isna(numeric_val):
                    # 按照要求，将 NaN、空值或非数字文本填充为 -1.0
                    tabular_data.append(-1.0)
                else:
                    # 否则，存储为浮点数
                    tabular_data.append(float(numeric_val))
            return tabular_data
        # --- 函数更新完毕 ---

        # --- 处理 Clinical (临床) 数据 ---
        print("Processing clinical_df...")
        exclude_cols_cli = ['cases.case_id', 'cases.submitter_id']
        tabular_cols_cli = [col for col in self.clinical_df.columns if col not in exclude_cols_cli]
        for idx, row in self.clinical_df.iterrows():
            patient_key = row['cases.submitter_id']
            case_id = row['cases.case_id']
            case_id_to_submitter[case_id] = patient_key
            
            # 提取表格数据
            self.clinical_tabular_dict[patient_key] = _process_row(row, tabular_cols_cli)

        # --- 处理 Genomics (基因组) 数据 ---
        print("Processing genomics_df...")
        self.genomics_tabular_dict = {}
        exclude_cols_gen = ['cases.case_id', 'cases.submitter_id']
        tabular_cols_gen = [col for col in self.genomics_df.columns if col not in exclude_cols_gen]
        for idx, row in self.genomics_df.iterrows():
            patient_key = row['cases.submitter_id']
            # 使用 get 和默认值 None 来安全处理
            if patient_key not in self.clinical_tabular_dict.get(patient_key, None): 
                self.genomics_tabular_dict[patient_key] = _process_row(row, tabular_cols_gen)

        # --- 处理 Histology/Pathology (病理) 数据 ---
        print("Processing histology_df...")
        self.pathology_tabular_dict = {}
        exclude_cols_path = ['cases.case_id', 'cases.submitter_id']
        tabular_cols_path = [col for col in self.histology_df.columns if col not in exclude_cols_path]
        for idx, row in self.histology_df.iterrows():
            patient_key = row['cases.submitter_id']
            # 使用 get 和默认值 None 来安全处理
            if patient_key not in self.clinical_tabular_dict.get(patient_key, None): 
                self.pathology_tabular_dict[patient_key] = _process_row(row, tabular_cols_path)

        # --- 处理 Reports (报告) 数据 ---
        print("Processing reports_df...")
        self.report_pathology = {}
        for idx, row in self.reports_df.iterrows():
            case_id = row['patient_id']
            if case_id not in case_id_to_submitter:
                continue
            patient_key = case_id_to_submitter[case_id]
            
            # 合并报告文本
            report_text = str(row["report_text"]) if pd.notna(row["report_text"]) else ""
            annotation_text = str(row["annotation_text"]) if pd.notna(row["annotation_text"]) else ""
            full_text = (report_text + " " + annotation_text).strip()
            
            if full_text:
                self.report_pathology[patient_key] = full_text

        # print tabular columns
        print("\n--- 检查表格数据维度 (Tabular Data Dimensions) ---")
        print(f"  [tabular-clinical]: {len(tabular_cols_cli)} columns.")
        print(f"  [tabular-genomics]: {len(tabular_cols_gen)} columns.")
        print(f"  [tabular-pathology]: {len(tabular_cols_path)} columns.")
        # [tabular-clinical]: 56 columns.
        # [tabular-genomics]: 27 columns.
        # [tabular-pathology]: 37 columns.
        print("-------------------------------------------------")
        print("请使用这些数字来设置 tcga_luad_pred.py 中的 'tabular_dims' 字典。")


        print("Finished processing all CSV files.")


    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        获取单个患者的数据。
        参考了 OSCCSurvInHouseDataset 的 __getitem__ 来实现生存标签逻辑。
        """
        # 1. 获取患者 ID (e.g., 'TCGA-LN-A49R')
        patient_id = self.patient_ids[idx]
        
        output_dict = {"pid": patient_id}

        # 2. 获取并处理生存标签 (来自 luad_sur_and_time.pkl)
        # --- 时间单位是 *月* ---
        try:
            survival_info = self.label_pickle[patient_id]
            event = int(survival_info[0])       # 0 = 审查 (censored), 1 = 事件 (death)
            time_months = float(survival_info[1]) # 生存时间（月）
        except KeyError:
            print(f"Error: Patient ID {patient_id} not found in label_pickle (luad_sur_and_time.pkl).")
            # 递归获取下一个，但要小心
            return self.__getitem__((idx + 1) % len(self))
        
        # c = 1 means censored, c = 0 means event occurred.
        # 我们的 TCGA event 标签是反的 (1=event), 所以我们转换它
        censorship = 1.0 - float(event)
        
        # 确保事件时间在 5 年 (60 个月) 内
        event_time = min(time_months, self.five_years_in_months)
        
        # 将连续时间离散化到 10 个 bins (0-9)
        # Y is the discrete time interval index
        time_bin = np.digitize(event_time, self.time_bins) - 1
        # 确保索引在 [0, num_bins-1]
        time_bin = max(0, min(time_bin, self.num_time_bins - 1))
        # --- BUG 修复结束 ---

        # --- 存储标签 ---
        output_dict['labels'] = {
            'label_Y': int(time_bin),      # 离散化的时间桶索引
            'label_c': int(censorship)     # 1=审查, 0=事件
        }
        output_dict['original_labels'] = {
            'label_Y': event_time,         # 连续时间 (截断到5年/60个月)
            'label_c': int(censorship)
        }

        # 3. 获取图数据 (来自 luad_data.pkl)
        try:
            # 这应该是一个 torch_geometric.data.Data 对象
            graph_data = copy.deepcopy(self.data_pickle[patient_id])
            output_dict["graph_data"] = graph_data
        except KeyError:
            print(f"Error: Patient ID {patient_id} not found in data_pickle (luad_data.pkl).")
            # 这是一个关键错误，这个 patient_id 没有图数据
            return self.__getitem__((idx + 1) % len(self))

        # 4. 根据 self.modalities 动态加载其他数据
        
        # --- (PyG) 图像特征 ---
        if "image-pathology" in self.modalities:
            # 图像特征已经包含在 graph_data.x_img 中
            output_dict["image-pathology"] = graph_data.x_img

        # --- (PyG) 基因组特征 ---
        if "genomics-genomics" in self.modalities:
            # 基因组特征已经包含在 graph_data.x_rna 中
            output_dict["genomics-genomics"] = graph_data.x_rna

        # --- 病理报告文本 ---
        if "text-pathology" in self.modalities:
            output_dict["text-pathology"] = self.report_pathology.get(patient_id, None)

        # --- 各种表格数据 ---
        if "tabular-clinical-56" in self.modalities:
            data = self.clinical_tabular_dict.get(patient_id, None)
            output_dict["tabular-clinical-56"] = torch.tensor(data, dtype=torch.float32) if data is not None else None

        if "tabular-genomics-27" in self.modalities:
            data = self.genomics_tabular_dict.get(patient_id, None)
            output_dict["tabular-genomics-27"] = torch.tensor(data, dtype=torch.float32) if data is not None else None
            
        if "tabular-pathology-37" in self.modalities:
            data = self.pathology_tabular_dict.get(patient_id, None)
            output_dict["tabular-pathology-37"] = torch.tensor(data, dtype=torch.float32) if data is not None else None

        # 5. 数据完整性检查 (可选，但推荐)
        # 确保至少加载了一种请求的模态，否则也跳过
        modalities_found = 0
        for mod_key in self.modalities:
            if mod_key in output_dict and output_dict[mod_key] is not None:
                # 特殊处理: 检查 tensor 是否为空 (尽管 _process_row 应该总是返回 -1)
                if isinstance(output_dict[mod_key], torch.Tensor):
                    if output_dict[mod_key].numel() > 0:
                        modalities_found += 1
                else:
                    modalities_found += 1
        
        if modalities_found == 0:
            # 没有找到任何请求的模态数据 (虽然 graph_data 总是会加载)
            # print(f"Warning: No *requested* modalities found for {patient_id}. Skipping.")
            # 递归不一定是个好主意，但为了匹配 OSCC 的逻辑，我们保留它
            return self.__getitem__((idx + 1) % len(self))

        return output_dict
    

if __name__ == "__main__":
    # 确保 __main__ 中的路径正确
    # 假设 Code 目录是: /home/Guanjq/NewWork/MedAlignFusion/Code
    # 那么 self.dataset_dir 将是: /home/Guanjq/NewWork/MedAlignFusion/Code/../Data/TCGA-LUAD
    # 看起来是正确的
    
    # 切换工作目录到 'Code' 目录 (如果需要)
    code_dir = "/home/Guanjq/NewWork/MedAlignFusion/Code"
    if os.path.exists(code_dir) and os.getcwd() != code_dir:
        try:
            os.chdir(code_dir)
            print(f"Changed working directory to: {os.getcwd()}")
        except Exception as e:
            print(f"Could not change directory to {code_dir}: {e}")
    else:
        print(f"Already in correct directory or {code_dir} not found.")


    print("Initializing dataset...")
    # 'all' 会加载所有已定义的模态
    dataset = TCGA_LUAD_Dataset(mode='train', modalities='all', fold=0)
    
    print(f"\n--- Total dataset length: {len(dataset)} ---")

    if len(dataset) > 0:
        print("\n--- Testing dataset[0] ---")
        data_item = dataset[0]
        print(f"Patient ID: {data_item['pid']}")
        
        print("\nLabels:")
        print(f"  Discrete (bin): {data_item['labels']['label_Y']}")
        print(f"  Censorship (1=censored, 0=event): {data_item['labels']['label_c']}")
        print(f"  Continuous (months): {data_item['original_labels']['label_Y']}")

        print("\nGraph Data:")
        print(f"  {data_item['graph_data']}")

        print("\nModalities:")
        for mod in dataset.modalities:
            if mod in data_item:
                data = data_item[mod]
                if isinstance(data, torch.Tensor):
                    print(f"  [{mod}]: torch.Tensor of shape {data.shape}")
                elif isinstance(data, str):
                    print(f"  [{mod}]: str of length {len(data)}")
                elif data is None:
                    print(f"  [{mod}]: None (Data not found for this patient)")
                else:
                    print(f"  [{mod}]: {type(data)}")
            else:
                 print(f"  [{mod}]: Not loaded (was not in data_item dict)")

        # 检查一个表格模态的示例值
        if "tabular-clinical-56" in data_item and data_item["tabular-clinical-56"] is not None:
            print("\nExample tabular-clinical-56 data (first 5 values):")
            print(f"  {data_item['tabular-clinical-56'][:5]}")
            # 检查是否有 -1.0
            if -1.0 in data_item["tabular-clinical-56"]:
                print("  -> Found -1.0 (imputed value) in the data.")
            else:
                print("  -> No -1.0 (imputed value) found in this sample.")

    print("\n--- Test Complete ---")