gpus=$1
feature_dim="128"
temperature=$2
k="200"
batch_size="$8"
# batch_size="128"
epochs="400"
estimator="$4"
dataset_name="$3"
tau_plus="$5"
beta="$6"
num_workers="$7"
# root_direc="../data"
seed="2023"

name1="NeurIPS2023_seed_${dataset_name}_${estimator}_t_${temperature}_eta1_${tau_plus}_eta2_${beta}_k_${k}_B_${batch_size}_Epoch_${epochs}"
echo $name1
CUDA_VISIBLE_DEVICES=$gpus python main.py --temperature $temperature --estimator $estimator --name $name1\
    --feature_dim $feature_dim --k $k --batch_size $batch_size \
    --tau_plus $tau_plus --num_workers $num_workers \
    --epochs $epochs --dataset_name $dataset_name --beta $beta --seed $seed
bash linear.sh $gpus $dataset_name $name1