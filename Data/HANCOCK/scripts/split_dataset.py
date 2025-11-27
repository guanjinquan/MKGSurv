import json
import random

random.seed(42)
save_file = "/home/Guanjq/NewWork/MedAlignFusion/Data/HANCOCK/DataSplits_DataDictionaries/dataset_split_train_valid_test.json"
input_file = "/home/Guanjq/NewWork/MedAlignFusion/Data/HANCOCK/DataSplits_DataDictionaries/dataset_split_treatment_outcome.json"


with open(input_file, "r") as f:
    data = json.load(f)


label_split_list = {}
for item in data:
    label = item['recurrent event or death']
    split = item['dataset']
    tuple = (label, split)
    if tuple not in label_split_list:
        label_split_list[tuple] = []
    label_split_list[tuple].append(item['patient_id'])

all_train = label_split_list[(1, 'training')] + label_split_list[(0, 'training')]

print("Length : ", len(label_split_list[(1, 'training')]))

valid_set = random.sample(label_split_list[(1, 'training')], 20)
train_set = [list(set(all_train) - set(valid_set))]
test_set = label_split_list[(1, 'test')]

for item in data:
    if item['patient_id'] in valid_set:
        item['dataset'] = 'valid'
    elif item['patient_id'] in test_set:
        item['dataset'] = 'test'
    else:
        item['dataset'] = 'train'


with open(save_file, "w") as f:
    json.dump(data, f, indent=4)