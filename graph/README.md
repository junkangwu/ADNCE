## Dependencies

* [PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric#installation)==1.6.0

Then, you need to create a directory for recoreding finetuned results to avoid errors:

```bash
mkdir logs
mkdir log_pos
```

## Training & Evaluation
Usage (example):
```bash
bash go_adnce.sh $GPU_ID $DATASET_NAME $AUGMENTATION $mode $w1 $w2 $temp
```

```$DATASET_NAME``` is the dataset name (please refer to https://chrsmrrs.github.io/datasets/docs/datasets/), ```$GPU_ID``` is the lanched GPU ID and ```$AUGMENTATION``` could be ```random2, random3, random4``` that sampling from {NodeDrop, Subgraph}, {NodeDrop, Subgraph, EdgePert} and {NodeDrop, Subgraph, EdgePert, AttrMask}, seperately. ```mode``` represents the model (easy or adnce). ```w1, w2, temp``` denote as mu, sigma and temperature respectively.

Commands for reproducing the reported results:

```bash
### NCI1
bash go_adnce.sh 0 NCI1 random3 easy 0.7 1.0 0.05
bash go_adnce.sh 0 NCI1 random3 adnce 0.7 1.0 0.05

### PROTEINS
bash go_adnce.sh 0 PROTEINS random4 easy 1.5 1.0 0.05
bash go_adnce.sh 0 PROTEINS random4 adnce 1.5 1.0 0.05

### DD
bash go_adnce.sh 0 DD random2 easy 0.2 1.0 0.2
bash go_adnce.sh 0 DD random2 adnce 0.2 1.0 0.2

### MUTAG
bash go_adnce.sh 0 MUTAG random2 easy 0.7 1.0 0.15
bash go_adnce.sh 0 MUTAG random2 adnce 0.7 1.0 0.15

### COLLAB
bash go_adnce.sh 0 COLLAB random4 easy 0.3 1.0 0.1
bash go_adnce.sh 0 COLLAB random4 adnce 0.3 1.0 0.1

### REDDIT-BINARY
bash go_adnce.sh 0 REDDIT-BINARY random2 easy 0.3 1.0 0.15
bash go_adnce.sh 0 REDDIT-BINARY random2 adnce 0.3 1.0 0.15

### REDDIT-MULTI-5K
bash go_adnce.sh 0 REDDIT-MULTI-5K random2 easy 0.8 1.0 0.15
bash go_adnce.sh 0 REDDIT-MULTI-5K random2 adnce 0.8 1.0 0.15

### IMDB-BINARY
bash go_adnce.sh 0 IMDB-BINARY random2 easy 0.7 1.0 0.55
bash go_adnce.sh 0 IMDB-BINARY random2 adnce 0.7 1.0 0.55
```
## Acknowledgements

The backbone implementation is reference to https://github.com/Shen-Lab/GraphCL/tree/master/unsupervised_TU.