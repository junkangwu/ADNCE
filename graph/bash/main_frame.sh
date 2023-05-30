#!/bin/bash
gpus="$1"
dataset="$2"
lr="0.01"
layers="3"
aug="$3"
mode="$4"
temp="$5"
w1="$6"
w2="$7"
num="$8"
intervel="$9"
char_name="${10}"
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
    if [[ $char_name == "w1" ]]
    then
        w1=`echo $w1 $intervel | awk '{ print $1+$2}'`
    elif [[ $char_name == "w2" ]]
    then
        w2=`echo $w2 $intervel | awk '{ print $1+$2}'`
    else
        temp=`echo $temp $intervel | awk '{ print $1+$2}'`
    fi
    # echo $intervel
    # echo $temp
done
