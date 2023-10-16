<h2 align="center">
Understanding Contrastive Learning via Distributionally Robust Optimization
</h2>

<!-- <div align="center"> -->

<!-- [![](https://img.shields.io/badge/paper-pink?style=plastic&logo=GitBook)](https://arxiv.org/pdf/2302.04775.pdf)
[![](https://img.shields.io/badge/-github-green?style=plastic&logo=github)](https://github.com/junkangwu/Adap_tau)  -->
<!-- [![](https://img.shields.io/badge/-youtube-red?style=plastic&logo=airplayvideo)](https://www.youtube.com/watch?v=9d0eXaO_kOw) 
[![](https://img.shields.io/badge/-slides-grey?style=plastic&logo=adobe)](https://cs.stanford.edu/~shirwu/slides/dir-iclr22.pdf)  -->
<!-- </div> -->

This is the PyTorch implementation for our NeurIPS 2023 paper. 
> Junkang Wu, Jiawei Chen, Jiancan Wu, Wentao Shi, Xiang Wang & Xiangnan He. 2023. Understanding Contrastive Learning via Distributionally Robust Optimization

# Pseudo Code
```py
# pos     : exp of inner products for positive examples
# neg     : exp of inner products for negative examples
# N       : number of negative examples
# t       : temperature scaling
# mu      : center position
# sigma   : height scale

#InfoNCE
standard_loss = -log(pos.sum() / (pos.sum() + neg.sum()))

#ADNCE
weight=1/(sigma * sqrt(2*pi)) * exp( -0.5 * ((neg-mu)/sigma)**2 )
weight=weight/weight.mean()
Adjusted_loss = -log(pos.sum() / (pos.sum() + (neg * weight.detach() ).sum()))
```


# Image Experiments
The code can be found in  [image](./image/README.md)

# Sentence Experiments
The code can be found in  [sentence](./sentence/README.md)

# Graph Experiments
The code can be found in  [graph](./graph/README.md)

# Citation
If you find this repo useful for your research, please consider citing the paper
```
@inproceedings{wu2023ADNCE,
  author       = {Junkang Wu and Jiawei Chen and Jiancan Wu and Wentao Shi and Xiang Wang and Xiangnan He},
  title        = {Understanding Contrastive Learning via Distributionally Robust Optimization},
  booktitle    = {NeurIPS},
  year         = {2023}
}
```


For any clarification, comments, or suggestions please create an issue or contact me (jkwu0909@gmail.com).