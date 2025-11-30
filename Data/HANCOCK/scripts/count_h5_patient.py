import os
import json

file = "/home/Guanjq/NewWork/MedAlignFusion/Data/HANCOCK/DataSplits_DataDictionaries/dataset_split_train_valid_test.json"

with open(file, 'r') as f:
    data = json.load(f)

test_set = [item['patient_id'] for item in data if item['dataset'] == 'test']

h5_root = "/home/Guanjq/NewWork/MedAlignFusion/Data/HANCOCK/WSI_UNI_encodings"

pat_set = set()
for root, dirs, files in os.walk(h5_root):
    for file in files:
        if file.endswith(".h5"):
            pat_id = file.split("_")[-1].split(".")[0]
            pat_set.add(pat_id)
            if pat_id in test_set:
                test_set.remove(pat_id)

print(test_set)
print(len(pat_set))