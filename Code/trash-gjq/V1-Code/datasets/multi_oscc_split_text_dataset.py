from torch.utils.data import Dataset, DataLoader, sampler
import numpy as np
import random
import os
import json
import torch
import torch.distributed as dist
from PIL import Image
from torchvision.transforms import functional as F
from torchvision.transforms import Compose, RandomVerticalFlip, RandomHorizontalFlip, RandomRotation, RandomAutocontrast, \
    RandomAdjustSharpness, RandomResizedCrop, Normalize, ToTensor, Resize
import pandas as pd 
from typing import List, Dict, Any



# ==========================================================================================
# Custom Transforms and Dataset Class
# ==========================================================================================

MEAN=[175.14728804175988, 110.57123792228117, 176.73598615775617]
STD=[21.239463551725915, 39.15991384752335, 10.99100631656543]
MEAN = [m / 255.0 for m in MEAN]
STD = [s / 255.0 for s in STD]


def TrainTransforms():
    """Returns a composition of transforms for training data augmentation."""
    return Compose([
        RandomResizedCrop(size=(512, 512), scale=(0.8, 1.0)),
        RandomVerticalFlip(p=0.5), 
        RandomHorizontalFlip(p=0.5),
        RandomRotation(degrees=(-45, 45)),
        RandomAutocontrast(p=0.5), 
        RandomAdjustSharpness(sharpness_factor=3, p=0.5),
        ToTensor(),
        Normalize(mean=MEAN, std=STD),
    ])

def InferTransforms():
    """Returns a composition of transforms for inference."""
    return Compose([
        Resize(size=(512, 512)),
        ToTensor(),
        Normalize(mean=MEAN, std=STD),
    ])


class MultiOSCCSplitDataset(Dataset):
    """
    Versión actualizada de Dataset que divide los datos clínicos en 5 modalidades de texto
    y carga dinámicamente los datos según las modalidades solicitadas.
    - Carga imágenes preprocesadas desde archivos .npy.
    - Omite pacientes a los que les faltan todas las modalidades solicitadas.
    """
    def __init__(self, mode="train", modalities="all"):
        super().__init__()
        assert mode in ["train", "valid", "test"], "mode must be one of 'train', 'valid', or 'test'"

        self.dataset_dir = os.path.join(os.getcwd(), "../Data/Multi-OSCCPI-Dataset")
        self.npy_dir = os.path.join(self.dataset_dir, "Multi-OSCCPI-Npy-512")

        # --- Variables miembro ---
        self.mode = mode
        self.items = []
        self.transforms = None
        self.clinical_df = None
        self.num_classes = 2
        self._column_check_done = False

        # --- MODIFICADO: Analizar y almacenar la lista de modalidades requeridas ---
        self.modalities = self._parse_modalities(modalities)
        print(f"Dataset will be initialized for modalities: {self.modalities}")

        # --- Lógica de inicialización ---
        self._load_clinical_data()
        self._load_and_filter_items() # Sin cambios desde _load_items

        if mode == "train":
            self.transforms = TrainTransforms()
        else:
            self.transforms = InferTransforms()

        if mode == "train":
            random.shuffle(self.items)

        print(f"Dataset loaded: mode='{self.mode}'. Final valid item count: {len(self.items)}")

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item_info = self.items[index]
        pid = item_info['pid']
        label = item_info['REC']

        # --- MODIFICADO: Construir dinámicamente el diccionario de salida ---
        output_dict = {}

        # --- Modalidad de imagen ---
        if "images" in self.modalities:
            npy_path = os.path.join(self.npy_dir, f"{pid}.npy")
            try:
                images_array = np.load(npy_path)
                assert images_array.shape[0] == 6, f"Expected 6 images, got {images_array.shape[0]} for PID {pid}."
                
                loaded_images = [Image.fromarray(images_array[i]) for i in range(images_array.shape[0])]
                
                if self.transforms:
                    transformed_images = [self.transforms(img) for img in loaded_images]
                else:
                    transformed_images = loaded_images
                
                output_dict["images"] = transformed_images

            except (FileNotFoundError, AssertionError) as e:
                print(f"Warning: NPY file missing or invalid for {pid} at getitem: {e}")
                pass # Omitir agregar la clave 'images'

        # --- MODIFICADO: Modalidades de texto ---
        # Comprobar si se solicita *alguna* modalidad de texto
        text_modalities_requested = any(m.startswith("text") for m in self.modalities)

        if text_modalities_requested:
            if self.clinical_df is not None and pid in self.clinical_df.index:
                patient_series = self.clinical_df.loc[pid]
                
                # Generar *todos* los 5 textos
                all_generated_texts = self._generate_clinical_text(patient_series)
                
                # Agregar solo los solicitados al diccionario de salida
                output_dict['text'] = []
                for key, text_content in all_generated_texts.items():
                    if key in self.modalities:
                        output_dict['text'].append(text_content)
                if len(output_dict['text']) > 0:
                    output_dict['text'] = " ".join(output_dict['text'])
                else:
                    output_dict['text'] = None
            else:
                output_dict['text'] = None

        # --- Etiquetas (siempre incluidas) ---
        one_hot_label = torch.zeros(self.num_classes, dtype=torch.float32) # Usar float para BCEWithLogitsLoss
        one_hot_label[label] = 1.0

        output_dict["pid"] = pid
        output_dict["labels"] = one_hot_label
        
        # --- Si no se cargaron modalidades con éxito, omitir este ítem ---
        # Contar claves de modalidad (excluyendo 'pid' y 'labels')
        modality_keys = set(output_dict.keys()) - {"pid", "labels"}
        count_not_none = sum(1 for key in modality_keys if output_dict[key] is not None)
        
        if count_not_none == 0:
             # Si no se cargó ninguna modalidad, intenta recursivamente con el siguiente ítem
             # Esto evita devolver un diccionario de datos vacío
             print(f"Warning: Skipping {pid} at index {index}, no valid modalities loaded.")
             return self.__getitem__((index + 1) % len(self))

        return output_dict  # return {"images", "text"}

    def __len__(self):
        return len(self.items)

    def _parse_modalities(self, modalities_str: str) -> List[str]:
        """
        MODIFICADO: Analiza la cadena de modalidades en una lista de claves de modalidad válidas.
        Las claves válidas ahora son 'image', 'text_1', 'text_2', 'text_3', 'text_4', 'text_5'.
        """
        if modalities_str == "all":
            return ["images", "text_1", "text_2", "text_3", "text_4", "text_5"]
        
        valid_set = {"images", "text_1", "text_2", "text_3", "text_4", "text_5"}
        
        # Permitir '-' y ',' como separadores
        parsed = [m.strip() for m in modalities_str.replace('-', ',').split(',')]
        
        for m in parsed:
            if m not in valid_set:
                raise ValueError(f"Invalid modality '{m}' specified. Must be one of {valid_set}")
        return parsed

    def _load_clinical_data(self):
        """Carga los datos clínicos CSV en un DataFrame de pandas."""
        clinical_data_path = os.path.join(self.dataset_dir, "clinical_data.csv")
        try:
            self.clinical_df = pd.read_csv(clinical_data_path)
            if 'PID' in self.clinical_df.columns:
                self.clinical_df.set_index('PID', inplace=True)
            print("Successfully loaded clinical data.")
        except FileNotFoundError:
            print(f"Warning: Clinical data file not found at {clinical_data_path}. Text modalities will be unavailable.")
            self.clinical_df = None # Asegurarse de que sea None si no se encuentra el archivo

    def _get_labels(self):
        """Devuelve una lista de todas las etiquetas en el dataset."""
        return [item['REC'] for item in self.items]
    
    def _check_modality_availability(self, pid: str) -> bool:
        """
        MODIFICADO: Comprueba si al menos una de las modalidades solicitadas está disponible
        para un paciente dado.
        """
        has_image = False
        if "images" in self.modalities:
            npy_path = os.path.join(self.npy_dir, f"{pid}.npy")
            has_image = os.path.exists(npy_path)

        has_text = False
        # Comprobar si se solicita *alguna* modalidad de texto
        if any(m.startswith("text_") for m in self.modalities):
            if self.clinical_df is not None and pid in self.clinical_df.index:
                has_text = True
        
        # El paciente es válido si *alguna* de las modalidades solicitadas está presente
        return has_image or has_text

    def _load_and_filter_items(self):
        """Carga los ítems del dataset y los filtra según la disponibilidad de la modalidad."""
        metadata_path = os.path.join(self.dataset_dir, "all_metadata.json")
        split_path = os.path.join(self.dataset_dir, "split_OOD.json")
        print("Load Split File:", split_path)

        with open(metadata_path, 'r') as f:
            all_patients_info = {item['pid']: item for item in json.load(f)['datainfo']}

        with open(split_path, 'r') as f:
            split_data = json.load(f)

        target_pids = set(split_data[self.mode])
        
        initial_items = [
            all_patients_info[pid] for pid in target_pids
            if pid in all_patients_info
        ]

        if len(initial_items) != len(target_pids):
            print(f"Warning: Some PIDs from split file not found in metadata. Found {len(initial_items)}/{len(target_pids)}.")

        # --- MODIFICADO: Filtrar ítems basados en la disponibilidad de la modalidad ---
        skipped_count = 0
        for item in initial_items:
            pid = item['pid']
            if self._check_modality_availability(pid):
                self.items.append(item)
            else:
                skipped_count += 1
        
        if skipped_count > 0:
            print(f"Skipped {skipped_count} patients because they were missing all requested modalities: {self.modalities}")

    def _generate_clinical_text(self, patient_series: pd.Series) -> Dict[str, str]:
        """
        MODIFICADO: Genera descripciones en lenguaje natural a partir de datos clínicos
        en 5 grupos separados basados en las listas de columnas proporcionadas.
        """
        
        # Definir los grupos de columnas basados en el JSON proporcionado
        column_groups = {
            "text_1": [
                "Gender(0male/1female)", "Age(Y)", "Weight(kg)", "Height(cm)", "AlcoholHistory(0no/1yes)",
                "SmokingHistory(0no/1yes)", "BetelNutHistory(0no/1yes)", "PreoperativeHistory(0no/1yes)",
                "PreoperativeHistoryDetails", "Diabetes(0no/1yes)", "RespiratoryDisease(0no/1yes)",
                "CardiovascularDisease(0no/1yes)", "MedControlledHypertension(0no/1yes)"
            ],
            "text_2": [
                "Pathology", "TumorT", "TumorN", "TumorM", "TumorLocation", "TumorDifferentiation(1high/2med/3low)",
                "CancerThrombus(0/1)", "SurroundingTissueInvasion(0/1)", "SurgicalMargin(0/1)", "LNM(0/1)",
                "IA(+)", "IB(+)", "IIA(+)", "IIB(+)", "III(+)", "NeckMass(+)", "AccessoryChain(+)",
                "VascularInvasion(+)", "PerineuralInvasion(+)"
            ],
            "text_3": [
                "SurgicalMethod", "SurgeryDuration", "ASAGrade", "NNISGrade", "IncisionType", "Flap",
                "NeckDissection", "Tracheotomy(0no/1yes)", "IntraoperativeBleeding(ml)", "PostopICU(0no/1yes)",
                "ICUDuration", "ReoperationWithin30d(0no/1yes)", "PostopComplications(0no/1yes)",
                "PostopComplicationDetails", "PostopDischargeTime(d)", "Deceased(0no/1yes)"
            ],
            "text_4": [
                "PreopWBC", "PreopHemoglobin", "PreopPotassium", "PreopAlbumin", "PreopVitaminD",
                "PostopWBC", "PostopHemoglobin", "PostopPotassium", "PostopAlbumin"
            ],
            "text_5": [
                "Ki-67", "CK5_6(0/1)", "P63(0/1)", "P16(0/1)", "HPV(0/1)", "PD_L1"
            ]
        }
        
        # --- AGREGADO: Verificación de aserción de una sola vez ---
        # Se ejecuta solo una vez para verificar que todas las columnas definidas
        # existan en el DataFrame.
        if not self._column_check_done:
            print("Running one-time column assertion check...")
            available_cols = set(patient_series.index)
            missing_cols = []
            
            all_defined_cols = [col for group_cols in column_groups.values() for col in group_cols]
            
            for col in all_defined_cols:
                if col not in available_cols:
                    missing_cols.append(col)
            
            assert not missing_cols, \
                f"Column mismatch error: The following columns were defined in " \
                f"column_groups but NOT found in the clinical data: {', '.join(missing_cols)}"
            
            self._column_check_done = True
            print("Column check passed successfully.")
        # --- FIN DE LA VERIFICACIÓN DE ASERCIÓN ---

        # Prefijos para cada grupo de texto
        prefixes = {
            "text_1": "Patient background and history: ",
            "text_2": "Tumor pathology and staging: ",
            "text_3": "Surgical and post-operative details: ",
            "text_4": "Pre-operative and post-operative lab results: ",
            "text_5": "Immunohistochemistry markers: "
        }

        # Función auxiliar para crear oraciones (sin cambios de tu original)
        def add_sentence(text_list, column_name, value):
            if pd.isna(value) or str(value).strip() in ['/', '']: return
            if isinstance(value, float) and value.is_integer(): value = int(value)
            sentence = ""
            
            # Lógica de mapeo de columnas (extendida para nuevas columnas)
            if column_name == "TumorT": sentence = f"The primary tumor stage (T stage) is {value}."
            elif column_name == "TumorN": sentence = f"The regional lymph node stage (N stage) is {value}."
            elif column_name == "TumorM": sentence = f"The distant metastasis stage (M stage) is {value}."
            elif column_name == "TumorDifferentiation(1high/2med/3low)":
                diff_map = {1: "well-differentiated", 2: "moderately-differentiated", 3: "poorly-differentiated"}
                sentence = f"The tumor differentiation is {diff_map.get(int(value), 'not specified')}."
            elif "(0/1)" in column_name or "(+)" in column_name:
                status = "present" if int(value) == 1 else "absent"
                feature_name = column_name.replace("(0/1)", "").replace("(+)", "").replace("_", " ")
                sentence = f"{feature_name} is {status}."
            elif "(0no/1yes)" in column_name:
                status = "yes" if int(value) == 1 else "no"
                feature_name = column_name.replace("(0no/1yes)", "").replace("History", " history")
                sentence = f"The patient has a record of {feature_name}: {status}."
            elif column_name == "Age(Y)": sentence = f"The patient's age is {value} years."
            elif column_name == "Gender(0male/1female)": sentence = f"The patient is {'female' if value == 1 else 'male'}."
            # Mapeo genérico para el resto
            elif column_name in [
                "Pathology", "SurgicalMethod", "TumorLocation", "Ki-67", "PD_L1",
                "Weight(kg)", "Height(cm)", "PreoperativeHistoryDetails", "SurgeryDuration",
                "ASAGrade", "NNISGrade", "IncisionType", "Flap", "NeckDissection",
                "IntraoperativeBleeding(ml)", "ICUDuration", "PostopComplicationDetails",
                "PostopDischargeTime(d)", "PreopWBC", "PreopHemoglobin", "PreopPotassium",
                "PreopAlbumin", "PreopVitaminD", "PostopWBC", "PostopHemoglobin",
                "PostopPotassium", "PostopAlbumin"
            ]:
                unit = ""
                col_name_simple = column_name
                if "(kg)" in column_name: unit = " kg"; col_name_simple = "Weight"
                elif "(cm)" in column_name: unit = " cm"; col_name_simple = "Height"
                elif "(ml)" in column_name: unit = " ml"; col_name_simple = "Intraoperative bleeding"
                elif "(d)" in column_name: unit = " days"; col_name_simple = "Postoperative discharge time"
                
                sentence = f"The {col_name_simple.lower().replace('_', ' ')} is recorded as: {value}{unit}."

            else:
                raise ValueError(f"Unrecognized column: {column_name}")

            if sentence: text_list.append(sentence)

        generated_texts = {}
        
        # Iterar a través de cada grupo de texto definido
        for key, columns in column_groups.items():
            sentences = []
            for col in columns:
                if col in patient_series:
                    add_sentence(sentences, col, patient_series[col])
            
            if sentences:
                generated_texts[key] = prefixes[key] + " ".join(sentences)
            else:
                generated_texts[key] = f"No detailed information available for {key.replace('_', ' ')}."

        return generated_texts