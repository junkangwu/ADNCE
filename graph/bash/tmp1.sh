gpus=$1
w2=$2
bash main_frame_robust.sh $gpus BZR random2 reweight 0.20 0.0 $w2 11 0.1 w1 0.1
bash main_frame_robust.sh $gpus BZR random2 reweight 0.50 0.0 $w2 11 0.1 w1 0.2
bash main_frame_robust.sh $gpus BZR random2 reweight 0.10 0.0 $w2 11 0.1 w1 0.3
bash main_frame_robust.sh $gpus BZR random2 reweight 0.10 0.0 $w2 11 0.1 w1 0.4
bash main_frame_robust.sh $gpus BZR random2 reweight 0.20 0.0 $w2 11 0.1 w1 0.5
bash main_frame_robust.sh $gpus BZR random2 reweight 0.40 0.0 $w2 11 0.1 w1 0.6