gpus=$1
dataset="$2"
model_file_name="$3"
pth_rectory="./results/${dataset}"
files=`ls ${pth_rectory}`
echo $files
for file in $files
do
    # echo $file
    if test -f "${pth_rectory}/${file}"
    then
        # if [[ $file == $model_file_name* ]]
        if [[ $file == $model_file_name* && "${file#*CNT}" == "_Epoch_400.pth" ]]
        then
            echo "Start to Evaluation"
            cnt_file="${pth_rectory}/${file}"
            echo $cnt_file
            CUDA_VISIBLE_DEVICES=$gpus python linear.py --dataset_name $dataset --model_path $cnt_file > ./outputs_/${file}.log
        else
            echo "NO NEED!!!"
        fi
    fi
done