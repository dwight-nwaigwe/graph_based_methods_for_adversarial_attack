source /applis/environments/cuda_env.sh 11.7
source ~/miniconda3/bin/activate
conda activate tf-gpu
python ~/scripts/create_data_wasserstein_mobilenet/WSR_make_data.py  9 
