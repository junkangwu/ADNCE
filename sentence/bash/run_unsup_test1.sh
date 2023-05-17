#!/bin/bash

# In this example, we show how to train SimCSE on unsupervised Wikipedia data.
# If you want to train it with multiple GPU cards, see "run_sup_example.sh"
# about how to use PyTorch's distributed data parallel.
gpus="$1"
loss_mode="$2"
w1="$3"
w2="$4"
model_name="bert_${loss_mode}_w1_${w1}_w2_${w2}"
echo $model_name
output_dir="result/${model_name}"
cd ..
CUDA_VISIBLE_DEVICES=$gpus python train.py \
    --model_name_or_path bert-base-uncased \
    --train_file data/wiki1m_for_simcse.txt \
    --output_dir $output_dir \
    --loss_mode $loss_mode \
    --w1 $w1 \
    --w2 $w2 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 64 \
    --learning_rate 3e-5 \
    --max_seq_length 32 \
    --evaluation_strategy steps \
    --metric_for_best_model stsb_spearman \
    --load_best_model_at_end \
    --eval_steps 125 \
    --pooler_type cls \
    --mlp_only_train \
    --overwrite_output_dir \
    --temp 0.05 \
    --do_train \
    --do_eval \
    --fp16 
