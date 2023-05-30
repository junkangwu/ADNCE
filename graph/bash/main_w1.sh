#!/bin/bash
gpus="$1"
dataset="$2"
lr="0.01"
layers="3"
aug="$3"
mode="$4"
w1="$5"
w2="$6"
temp="$7"
num="$8"
for ((i=1; i<=$num; i++))
do
    for seed in 0 1 2 3 4
    do
        cd ..
        model_name="${dataset}_seed_${seed}_${aug}_mode_${mode}_t_${temp}_w1_${w1}_w2_${w2}"
        echo $model_name
        CUDA_VISIBLE_DEVICES=$gpus python gsimclr_robust.py --DS $dataset --lr $lr --local --num-gc-layers $layers --aug $aug --seed $seed --mode $mode \
        --temp $temp --w1 $w1 --w2 $w2 --model_name $model_name
        cd bash
    done
    w1=`echo $w1 | awk '{ print $1+0.1}'`
done
