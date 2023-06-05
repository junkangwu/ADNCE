## Requirements

First, install PyTorch by following the instructions from [the official website](https://pytorch.org). To faithfully reproduce our results, please use the correct `1.7.1` version corresponding to your platforms/CUDA versions. PyTorch version higher than `1.7.1` should also work. For example, if you use Linux and **CUDA11** ([how to check CUDA version](https://varhowto.com/check-cuda-version/)), install PyTorch by the following command,

```bash
pip install torch==1.7.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html
```

If you instead use **CUDA** `<11` or **CPU**, install PyTorch by the following command,

```bash
pip install torch==1.7.1
```


Then run the following script to install the remaining dependencies,

```bash
pip install -r requirements.txt
mkdir logs
```

## Training and Evaluation
Usage (example):
```bash
bash main_w1_frame.sh $GPU_ID $mode $w1 $w2 $temp $model_name $bsz $lr
```

```$GPU_ID``` is the lanched GPU ID and ```mode``` represents the model (easy or adnce). ```w1, w2, temp``` denote as mu, sigma and temperature respectively. ```$model_name``` refers to BERT_base or RoBERTa_base. ```$bsz``` and ```$lr``` are hyperparamters for training:

Commands for reproducing the reported results:

```bash
### BERT_base
bash main_frame.sh 0 adnce 0.4 1.0 0.07 bert-base-uncased 64 3e-5

### RoBERTa_base
bash main_frame.sh 0 adnce 2.0 1.0 0.06 RoBERTa-base 512 1e-5
```

## Acknowledgements

The backbone implementation is reference to https://github.com/princeton-nlp/SimCSE.