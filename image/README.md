
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
bash train_test.sh $GPU_ID $tmp $DATASET_NAME $mode $w1 $w2 $num_works $bsz
```

```$GPU_ID``` is the lanched GPU ID and ```w1, w2, temp``` denote as mu, sigma and temperature respectively. ```$DATASET_NAME``` is the dataset name (cifar10, cifar100, stl10),  ```mode``` represents the model (easy or adnce). ```$num_works``` and ```$bsz``` are hyperparamters for training:

Commands for reproducing the reported results:

```bash
### CIFAR10
bash train_test.sh 0 0.4 cifar10 adnce 0.7 0.5 8 256

### CIFAR100
bash train_test.sh 0 0.3 cifar100 adnce 0.5 1.0 8 256

### STL10
bash train_test.sh 0 0.2 stl10 adnce 0.8 1.0 8 256

```

## Acknowledgements

Part of this code is inspired by [leftthomas/SimCLR](https://github.com/leftthomas/SimCLR) and [chingyaoc/DCL](https://github.com/chingyaoc/DCL).
