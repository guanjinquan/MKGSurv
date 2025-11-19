import os
import json
from typing import Dict, Any, List, Tuple
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import random
import copy
import joblib # 导入 joblib
from sklearn.cluster import KMeans # <-- 新增导入
from datasets.dataset_base import MultiModalDataset
from modules.common_modules.prototypes import cluster



class TCGA_LUSC_Dataset(MultiModalDataset):

    TREATMENT_OPTIONS = None

    PRE_OP_MODALITIES = [
        "tabular-clinical-9", 
        "genomics-genomics",
        "image-pathology", 
    ]

    POST_OP_MODALITIES = [
        "text-pathology", 
        "text-treatment",
        "tabular-treatment-8", 
    ]

    VALID_MODALITIES = PRE_OP_MODALITIES + POST_OP_MODALITIES


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
        self.fold = fold

        """
        Data(x_img=[633, 1024], x_rna=[5, 1024], x_cli=[11, 1024],  data_type=[3], edge_index_image=[2, 4606], edge_index_rna=[2, 20], edge_index_cli=[2, 110])
        如何访问 (示例): data.x_img (访问图像特征)
                              data.edge_index_rna (访问RNA边)
                              data.sur_type (访问生存类型)
        """
        self.dataset_dir = os.path.join(os.getcwd(), "../Data/TCGA-LUSC") 
        print(f"Warning: Using dataset directory: {self.dataset_dir}")
        print("Please ensure this path is correct or modify as needed.")

        # Load Image Features
        self.data_pickle = self._read_pickle(os.path.join(self.dataset_dir, "source", "lusc_data.pkl")) # DICT ID:torch_geometric.data.data.Data, 

        # Load Gene Features
        gene_pkl_file = os.path.join(self.dataset_dir, "processed", "hallmarks_tokens_pid_map.pkl")
        self.gene_dict = self._read_pickle(gene_pkl_file)  # PID -> Tokens

        # 加载 5 折交叉验证数据
        # [fold] 会选择是第 0, 1, 2, 3, 还是 4 折
        split_file = os.path.join(self.dataset_dir, "processed", "lusc_patients_5fold.json") 
        with open(split_file, 'r') as f:
            self.patient_ids = json.load(f)['folds'][fold][mode]

        # --- Parse modalities ---
        self.modalities = self.parse_modalities(modalities)
        print(f"Dataset will be initialized for modalities: {self.modalities}")

        # --- Load and preprocess all data sources ---
        self._load_data()
        print(f"Dataset for mode '{self.mode}' initialized. Found {len(self.patient_ids)} patients.")

    def _load_data(self):
        # 定义 processed 目录的路径
        processed_dir = os.path.join(self.dataset_dir, "processed")
        print(f"Loading data from processed directory: {processed_dir}")

        # 定义文件路径
        clinical_path = os.path.join(processed_dir, "clinical_data_aggregated.csv")
        treatment_path = os.path.join(processed_dir, "treatment_data_aggregated.csv")

        reports_path = os.path.join(processed_dir, "tcga_lusc_reports.csv")
        labels_path = os.path.join(processed_dir, "lusc_patient_labels.csv")  
        
        try:
            # --- 更新：加载时明确指定 dtype=str，以防止 pandas 自动转换 ---
            # 这样可以确保 '1.0' 和 '1' 都被视为字符串，直到 _process_row 处理
            self.clinical_df = pd.read_csv(clinical_path, dtype=str)
            print(f"Loaded clinical data with shape: {self.clinical_df.shape}")
            self.treatment_df = pd.read_csv(treatment_path, dtype=str)
            print(f"Loaded treatment data with shape: {self.treatment_df.shape}")
            self.reports_df = pd.read_csv(reports_path, dtype=str) 
            print(f"Loaded reports data with shape: {self.reports_df.shape}")
            self.labels_df = pd.read_csv(labels_path, dtype=str) 
            print(f"Loaded reports data with shape: {self.labels_df.shape}")
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
            columns = sorted(columns)
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

        # --- 处理 Treatment 数据 ---
        self.treatment_tabular_dict = {}
        tabular_cols_treat = [col for col in self.treatment_df.columns if col not in exclude_cols_cli]
        for idx, row in self.treatment_df.iterrows():
            patient_key = row['cases.submitter_id']
            case_id = row['cases.case_id']
            case_id_to_submitter[case_id] = patient_key
            
            # 提取表格数据
            self.treatment_tabular_dict[patient_key] = _process_row(row, tabular_cols_treat)

        # --- 处理 Reports (报告) 数据 ---
        print("Processing reports_df...")
        self.report_pathology = {}
        for idx, row in self.reports_df.iterrows():
            patient_key = row['patient_id'].strip()

            # 合并报告文本
            report_text = str(row["report_text"]) if pd.notna(row["report_text"]) else ""
            annotation_text = str(row["annotation_text"]) if pd.notna(row["annotation_text"]) else ""
            full_text = (report_text + " " + annotation_text).strip()
            
            if full_text:
                self.report_pathology[patient_key] = full_text
            else:
                print(f"No found report for patient {patient_key}")

        # --- 处理标签 ---
        # Load treatments.treatment_type
        self.TREATMENT_OPTIONS = []
        self.patient_labels = {}
        for idx, row in self.labels_df.iterrows():
            patient_key = row['cases.submitter_id']
            case_id = row['cases.case_id']
            self.patient_labels[patient_key] = {
                "DFS_time": float(row["DFS_time"]),
                "DFS_event": float(row["DFS_event"]),
                "Treatment_type": str(row['treatments.treatment_type']),
                "Treatment_type_id": str(row["5_classes"])
            }
            self.TREATMENT_OPTIONS.append(str(row['treatments.treatment_type']))

        self.TREATMENT_OPTIONS = list(set(self.TREATMENT_OPTIONS))
        print("Load self.TREATMENT_OPTIONS length = ", len(self.TREATMENT_OPTIONS))

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

        # 2. 获取并处理生存标签 
        # --- 时间单位是 *days* ---
        try:
            survival_info = self.patient_labels[patient_id]
            event = int(survival_info['DFS_event'])      # 0 = 审查 (censored), 1 = 事件 (death)
            time_days = float(survival_info['DFS_time'])    # 生存时间（days）
            treatment = str(survival_info['Treatment_type'])
            treatment_type_id = str(survival_info['Treatment_type_id'])
        except Exception as e:
            print(f"Error: {e}")
            return self.__getitem__((idx + 1) % len(self))
        
        output_dict['labels'] = {
            'label_time': time_days,
            'label_event': event,      # 1 means event happen!!, 0 means censored
            'treatment_type': treatment,
            'treatment_type_onehot': [1 if str(i) in treatment_type_id else 0 for i in range(5) ],
        }

        # 3. 获取图数据  
        try:
            # 这应该是一个 torch_geometric.data.Data 对象
            graph_data = copy.deepcopy(self.data_pickle[patient_id])
            output_dict["graph_data"] = graph_data
        except KeyError:
            print(f"Error: Patient ID {patient_id} not found in data_pickle")
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
            output_dict["genomics-genomics"] = self.gene_dict[patient_id]

        # --- 病理报告文本 ---
        if "text-pathology" in self.modalities:
            output_dict["text-pathology"] = self.report_pathology.get(patient_id, None)
            if patient_id not in self.report_pathology:
                print(f"Not found report for patient : {patient_id}")

        if "text-treatment" in self.modalities:
            output_dict["text-treatment"] = treatment

        # --- 各种表格数据 ---
        if "tabular-clinical-9" in self.modalities:
            data = self.clinical_tabular_dict.get(patient_id, None)
            output_dict["tabular-clinical-9"] = torch.tensor(data, dtype=torch.float32) if data is not None else None

        if "tabular-treatment-8" in self.modalities:
            data = self.treatment_tabular_dict.get(patient_id, None)
            output_dict["tabular-treatment-8"] = torch.tensor(data, dtype=torch.float32) if data is not None else None

        # 5. 数据完整性检查 
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
            return self.__getitem__((idx + 1) % len(self))

        if self.mode == 'train' and random.random() < 0.5:
            output_dict = self.aug_treatment_data(output_dict)

        return output_dict
    
    def aug_treatment_data(self, output_dict):
        treatment_keys = [
            "text-pathology", 
            "text-treatment",
            "tabular-treatment-8", 
        ]

        def aug_tabular(tabular_data):
            for i in range(len(tabular_data)):
                tabular_data[i] += random.gauss() * 0.1
            return tabular_data
        
        def aug_text(text):
            # random shuffle after split '.' or '+'
            for tag in ['.', '+']:
                text_parts = text.split(tag)
                random.shuffle(text_parts)
                text = tag.join(text_parts)
            return text.strip()

        for key in treatment_keys:
            if key not in output_dict:
                continue

            # 20% possibility to drop!
            random_prob = random.random()
            if random_prob < 0.5:
                if sum([1 for val in output_dict.values() if val is not None]) > 3 and len(self.modalities) > 1:  # labels, pids, + one column
                    output_dict[key] = None

            else:
                # process text
                if output_dict[key] is not None:
                    if "text" in key:
                        output_dict[key] = aug_text(output_dict[key])
                    elif "tabular" in key:
                        output_dict[key] = aug_tabular(output_dict[key])

        return output_dict
    
    def get_survival_bins(self):
        """
        Returns a list of all labels (time bins) in the dataset.
        This is used by the SurvivalBalancedBatchSampler.
        """
        self.observed_years = 20 * 365.0
        self.num_time_bins = 20
        self.time_bins = np.linspace(0, self.observed_years, self.num_time_bins + 1)

        labels_y = []
        for patient_id in self.patient_ids:
            try:
                survival_info = self.patient_labels[patient_id]
                time_days = float(survival_info['DFS_time']) 

                event_time = min(time_days, self.observed_years)
                time_bin = np.digitize(event_time, self.time_bins) - 1
                
                # 确保索引在 [0, num_bins-1]
                time_bin = max(0, min(time_bin, self.num_time_bins - 1))
                
                labels_y.append(int(time_bin))

            except Exception as e:
                print(f"Error processing label for {patient_id}: {e}. Assigning time_bin = 0 (default).")
                labels_y.append(0) # 附加一个默认 bin (0)
                
        return labels_y

    def parse_modalities(self, modalities_str: str) -> List[str]:
        """Parses the modalities string into a list of valid modality keys."""

        if modalities_str == "all":
            # 返回所有有效的模态
            return sorted(list(self.VALID_MODALITIES))
        
        # 解析逗号分隔的字符串
        requested_modalities = modalities_str.split(',')
        parsed_list = []
        for mod in requested_modalities:
            mod = mod.strip() # 清理空格
            if mod in self.VALID_MODALITIES:
                parsed_list.append(mod)
            else:
                raise ValueError(f"Warning: Modality '{mod}' not recognized and will be skipped.")
        
        return parsed_list

    def get_active_modalities(self):
        return self.modalities
    
    
    def get_training_image_embeddings_prototypes(self, num_prototypes=64):
        cache_dir = os.path.join(os.getcwd(), "../Cache")
        cache_file = os.path.join(cache_dir, f"TCGA_LUAD_Fold={self.fold}_Prototypes={num_prototypes}.npy")
        
        # 检查缓存是否存在
        if os.path.exists(cache_file):
            print(f"Loading cached prototypes from: {cache_file}")
            try:
                prototypes_np = np.load(cache_file)
                # 验证加载的原型数量是否匹配
                if prototypes_np.shape[0] == num_prototypes:
                    print("Cache hit. Returning cached prototypes.")
                    return prototypes_np # 返回 NumPy 数组
                else:
                    print(f"Cache mismatch. Expected {num_prototypes} prototypes, found {prototypes_np.shape[0]}. Recalculating...")
            except Exception as e:
                print(f"Error loading cache file {cache_file}: {e}. Recalculating...")

        all_embeddings = []
        print(f"Aggregating image embeddings from {len(self)} patients in '{self.mode}' split...")

        for idx in range(len(self)):
            patient_id = self.patient_ids[idx]
            try:
                # 注意: 无需 deepcopy，因为我们只读取数据
                graph_data = self.data_pickle[patient_id]
            except KeyError:
                print(f"Warning: Patient ID {patient_id} not found in data_pickle during prototype creation. Skipping.")
                continue
            
            # 检查 x_img 是否存在、非空且有数据
            if hasattr(graph_data, 'x_img') and graph_data.x_img is not None and graph_data.x_img.shape[0] > 0:
                all_embeddings.append(graph_data.x_img.cpu().numpy())

        if not all_embeddings:
            print("Error: No image embeddings found to cluster. Returning None.")
            return None

        # 将所有找到的嵌入连接成一个大的 NumPy 数组
        try:
            all_embeddings_np = np.concatenate(all_embeddings, axis=0)
        except ValueError as e:
            print(f"Error concatenating embeddings: {e}. Check if all x_img have the same feature dimension.")
            return None

        print(f"Total image patches (embeddings) found: {all_embeddings_np.shape[0]}")

        # 处理边角情况: 嵌入数量少于要求的原型数量
        if all_embeddings_np.shape[0] < num_prototypes:
            print(f"Warning: Found only {all_embeddings_np.shape[0]} embeddings, which is less than num_prototypes ({num_prototypes}).")
            print("Returning the unique embeddings themselves as prototypes.")
            # 使用 np.unique 确保返回的至少是唯一的嵌入
            unique_embeddings = np.unique(all_embeddings_np, axis=0)
            print("This edge-case result will not be cached.")
            return unique_embeddings # <-- 返回 NumPy 数组

        # 将 NumPy 数组转换为 PyTorch Tensor (cluster 函数期望 Tensor)
        all_embeddings_tensor = torch.from_numpy(all_embeddings_np).float()

        try:
            # 调用导入的 cluster 函数
            prototypes_with_batch_dim = cluster(
                patches=all_embeddings_tensor,
                n_proto=num_prototypes
            )
            
            # cluster 函数返回 (1, K, D) 的 NumPy 数组
            # 我们的缓存逻辑期望 (K, D)，所以执行 squeeze
            prototypes_np = prototypes_with_batch_dim.squeeze(0)
            
            print("Clustering complete.")

            try:
                os.makedirs(cache_dir, exist_ok=True)
                np.save(cache_file, prototypes_np)
                print(f"Saved prototypes to cache: {cache_file}")
            except Exception as e:
                print(f"Warning: Could not save cache file to {cache_file}: {e}")

            return prototypes_np   # <-- 返回 NumPy 数组
        
        except Exception as e:
            return None


if __name__ == "__main__":
    # 确保 __main__ 中的路径正确
    # 假设 Code 目录是: /home/Guanjq/NewWork/MedAlignFusion/Code
    # 那么 self.dataset_dir 将是: /home/Guanjq/NewWork/MedAlignFusion/Code/../Data/TCGA-LUAD
    # 看起来是正确的
    
    # 切换工作目录到 'Code' 目录 (如果需要)
    code_dir = "/home/Guanjq/NewWork/MedAlignFusion/Code"
    os.chdir(code_dir)

    dataset = TCGA_LUSC_Dataset(fold=0)

    data = dataset.get_training_image_embeddings_prototypes()
