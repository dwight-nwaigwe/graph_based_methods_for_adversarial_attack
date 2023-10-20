source ~/miniconda3/bin/activate
source /applis/environments/cuda_env.sh 11.7
conda activate tf-gpu
python ~/scripts/create_data_mobilenet/create_save_nodestats.py 6000 7000

python C:\\Users\\dwight\\Desktop\\scripts\\mobilenet\\logistic_regression_on_graph_data\create_data_mobilenet\\create_save_nodestats.py 0 1000
