conda env create -f  BuildEnv/env.yaml
source activate medfusion
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r BuildEnv/requirements.txt

