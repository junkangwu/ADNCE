#!/bin/bash
gpus="$1"
loss_mode="$2"
w1="$3"
w2="$4"
model_name="bert_${loss_mode}_w1_${w1}_w2_${w2}"
echo $model_name
output_dir="result/${model_name}"
bash run_unsup_test1.sh $gpus $loss_mode $w1 $w2
cd bash
bash evaluate.sh $gpus $model_name
