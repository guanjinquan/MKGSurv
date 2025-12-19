import os
from huggingface_hub import snapshot_download

os.chdir(os.path.dirname(__file__))

model_name = "MahmoodLab/CONCH"
save_dir = "/home/Guanjq/NewWork/MedAlignFusion/PretrainedWeights"

while True:
    try:
        print("Downloading Model...", model_name, flush=True)
        snapshot_download(
            model_name,
            local_dir=f"{save_dir}/{model_name}",
        )
        print("Model downloaded successfully.", flush=True)
        break
    except Exception as e:
        print(f"Error occurred: {e}. Retrying...", flush=True)
