
## Prerequisites
- Python 3.7 
- PyTorch 1.3.1
- PIL
- OpenCV

## Training & Evaluation
Usage (example):
```bash
mkdir logs
mkdir outputs_
bash train_test.sh $GPU_ID $tmp $DATASET_NAME $mode $w1 $w2 $num_worls $bsz
```

```$GPU_ID``` is the lanched GPU ID and ```w1, w2, temp``` denote as mu, sigma and temperature respectively. ```$DATASET_NAME``` is the dataset name (cifar10, cifar100, stl10),  ```mode``` represents the model (easy or adnce). ```$num_worls``` and ```$bsz``` are hyperparamters for training:


## Acknowledgements

Part of this code is inspired by [leftthomas/SimCLR](https://github.com/leftthomas/SimCLR) and [chingyaoc/DCL](https://github.com/chingyaoc/DCL).
