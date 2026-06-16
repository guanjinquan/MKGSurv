conda env create --file  ./env.yaml
source activate mkgsurv
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
conda install -c pytorch faiss-gpu
pip install -r BuildEnv/requirements.txt
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.1+cu121.html
pip install torch_geometric

