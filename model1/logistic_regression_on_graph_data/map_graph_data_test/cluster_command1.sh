source ~/miniconda3/bin/activate
conda activate tf-gpu
source /applis/environments/cuda_env.sh 11.7
python ./map_graph_data_test.py "edges"
