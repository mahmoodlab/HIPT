
```python
GPU=0
TASK=tcga_brca_subtype
DATAROOT='/home/richard/Richard/HIPT/'
CUDA_VISIBLE_DEVICES=$GPU python main.py --data_root_dir $DATAROOT --model_type mil --task $TASK --mode local_region_features --prop 1.0 --path_input_dim 192
CUDA_VISIBLE_DEVICES=$GPU python main.py --data_root_dir $DATAROOT --model_type hipt_gp --task $TASK --mode local_region_features --prop 1.0 --path_input_dim 192
GPU=0
TASK=tcga_brca_subtype
DATAROOT='/home/richard/Richard/HIPT/'
CUDA_VISIBLE_DEVICES=$GPU python main.py --data_root_dir $DATAROOT --model_type mil --task $TASK --mode local_region_features --prop 0.25 --path_input_dim 192
CUDA_VISIBLE_DEVICES=$GPU python main.py --data_root_dir $DATAROOT --model_type hipt_gp --task $TASK --mode local_region_features --prop 0.25 --path_input_dim 192

CUDA_VISIBLE_DEVICES=0 python main.py --drop_out --exp_code brca_mil --task tcga_brca_subtype --model_type mil --log_data --data_root_dir /home/richard/Richard/HIPT/ --split_dir tcga_brca
CUDA_VISIBLE_DEVICES=1 python main.py --drop_out --exp_code brca_hipt --task tcga_brca_subtype --model_type hipt_gp --log_data --data_root_dir /home/richard/Richard/HIPT/ --split_dir tcga_brca

GPU=1
TASK=tcga_lung_subtype
DATAROOT='/home/richard/Richard/HIPT/'
CUDA_VISIBLE_DEVICES=$GPU python main.py --data_root_dir $DATAROOT --model_type mil --task $TASK --mode local_region_features --prop 1.0 --path_input_dim 192
CUDA_VISIBLE_DEVICES=$GPU python main.py --data_root_dir $DATAROOT --model_type hipt_gp --task $TASK --mode local_region_features --prop 1.0 --path_input_dim 192
GPU=1
TASK=tcga_lung_subtype
DATAROOT='/home/richard/Richard/HIPT/'
CUDA_VISIBLE_DEVICES=$GPU python main.py --data_root_dir $DATAROOT --model_type mil --task $TASK --mode local_region_features --prop 0.25 --path_input_dim 192
CUDA_VISIBLE_DEVICES=$GPU python main.py --data_root_dir $DATAROOT --model_type hipt_gp --task $TASK --mode local_region_features --prop 0.25 --path_input_dim 192


GPU=2
TASK=tcga_kidney_subtype
DATAROOT='/home/richard/Richard/HIPT/'
CUDA_VISIBLE_DEVICES=$GPU python main.py --data_root_dir $DATAROOT --model_type mil --task $TASK --mode local_region_features --prop 1.0 --path_input_dim 192
CUDA_VISIBLE_DEVICES=$GPU python main.py --data_root_dir $DATAROOT --model_type hipt_gp --task $TASK --mode local_region_features --prop 1.0 --path_input_dim 192
GPU=2
TASK=tcga_kidney_subtype
DATAROOT='/home/richard/Richard/HIPT/'
CUDA_VISIBLE_DEVICES=$GPU python main.py --data_root_dir $DATAROOT --model_type mil --task $TASK --mode local_region_features --prop 0.25 --path_input_dim 192
CUDA_VISIBLE_DEVICES=$GPU python main.py --data_root_dir $DATAROOT --model_type hipt_gp --task $TASK --mode local_region_features --prop 0.25 --path_input_dim 192
```