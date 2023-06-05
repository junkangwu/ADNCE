#!/bin/bash
gpus="$1"
model_name="$2"
model_name_or_path="./result/${model_name}"
CUDA_VISIBLE_DEVICES=$gpus python evaluation.py \
    --model_name_or_path $model_name_or_path \
    --pooler cls \
    --task_set sts \
    --mode test > ./logs/${model_name}.log
