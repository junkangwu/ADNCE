#!/bin/bash
gpus="$1"
loss_mode="$2"
w1="$3"
w2="$4"
temp="$5"
model_name_or_path="$6"
model_name="${model_name_or_path}_${loss_mode}_t_${temp}_w1_${w1}_w2_${w2}"
echo $model_name
output_dir="result/${model_name}"
bash run_unsup_test0.sh $gpus $loss_mode $w1 $w2 $temp $model_name $model_name_or_path
cd bash
bash evaluate.sh $gpus $model_name
