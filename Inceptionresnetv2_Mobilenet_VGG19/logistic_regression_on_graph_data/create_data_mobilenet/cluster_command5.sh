source ~/miniconda3/bin/activate
source /applis/environments/cuda_env.sh 11.7
conda activate tf-gpu
python ~/scripts/create_data_mobilenet/create_save_nodestats.py 5000 6000

