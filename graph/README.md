## Dependencies

* [PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric#installation)==1.6.0

Then, you need to create a directory for recoreding finetuned results to avoid errors:

```
mkdir logs
```

## Training & Evaluation

```
./go.sh $GPU_ID $DATASET_NAME $AUGMENTATION
```sh

```$DATASET_NAME``` is the dataset name (please refer to https://chrsmrrs.github.io/datasets/docs/datasets/), ```$GPU_ID``` is the lanched GPU ID and ```$AUGMENTATION``` could be ```random2, random3, random4``` that sampling from {NodeDrop, Subgraph}, {NodeDrop, Subgraph, EdgePert} and {NodeDrop, Subgraph, EdgePert, AttrMask}, seperately.

## Acknowledgements

The backbone implementation is reference to https://github.com/fanyun-sun/InfoGraph/tree/master/unsupervised.

# REDDIT-BINARY NCI1 PROTEINS DD
# bash go.sh 7 REDDIT-BINARY random2
# bash go.sh 5 NCI1 random2
bash go_robust.sh 1 COLLAB random4 adnce 1.0
bash go_robust.sh 0 NCI1 random2 graphcl_robust 1e-3
bash go_robust.sh 2 NCI1 random2 graphcl_robust 1e-4
bash go_robust.sh 3 NCI1 random2 graphcl_robust 1e-5

============================================================
COLLAB_easy         	0.7226	0.011	COLLAB_random4_mode_easy_t_0.10_w1_1.0_w2_1.0
COLLAB_reweight     	0.7194	0.014	COLLAB_random4_mode_reweight_t_0.10_w1_0.3_w2_1.0
DD_easy             	0.7846	0.0095	DD_random2_mode_easy_t_0.05_w1_1.0_w2_1.0
DD_reweight         	0.7923	0.0059	DD_random2_mode_reweight_t_0.20_w1_0.2_w2_1.0
IMDB-BINARY_easy    	0.7172	0.0061	IMDB-BINARY_random2_mode_easy_t_0.55_w1_1.0_w2_1.0
IMDB-BINARY_reweight	0.7158	0.0072	IMDB-BINARY_random2_mode_reweight_t_0.55_w1_0.7_w2_1.0
MUTAG_easy          	0.8841	0.0087	MUTAG_random2_mode_easy_t_0.15_w1_1.0_w2_1.0
MUTAG_reweight      	0.8904	0.013	MUTAG_random2_mode_reweight_t_0.15_w1_0.7_w2_1.0
NCI1_easy           	0.7926	0.0034	NCI1_random3_mode_easy_t_0.05_w1_1.0_w2_1.0
NCI1_reweight       	0.793	0.0067	NCI1_random3_mode_reweight_t_0.05_w1_0.7_w2_1.0
PROTEINS_easy       	0.7507	0.0059	PROTEINS_random4_mode_easy_t_0.05_w1_1.0_w2_1.0
PROTEINS_reweight   	0.7477	0.0063	PROTEINS_random4_mode_reweight_t_0.05_w1_1.5_w2_1.0
REDDIT-BINARY_easy  	0.9073	0.0064	REDDIT-BINARY_random2_mode_easy_t_0.15_w1_1.0_w2_1.0
REDDIT-BINARY_reweight	0.9139	0.0031	REDDIT-BINARY_random2_mode_reweight_t_0.15_w1_0.3_w2_1.0
REDDIT-MULTI-5K_easy	0.5583	0.0034	REDDIT-MULTI-5K_random2_mode_easy_t_0.15_w1_1.0_w2_1.0
REDDIT-MULTI-5K_reweight	0.5601	0.0035	REDDIT-MULTI-5K_random2_mode_reweight_t_0.15_w1_0.8_w2_1.0
============================================================
