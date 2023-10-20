source /applis/environments/cuda_env.sh 11.7
source ~/miniconda3/bin/activate
conda activate tf-gpu
python ./map_graph_data.py 'outin'
