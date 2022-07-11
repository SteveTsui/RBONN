# Recurrent Bilinear Optimization for Binary Neural Networks (RBONN)
Pytorch implementation of RBONN in ECCV 2022.
## Tips

Any problem, please contact the first author (Email: shengxu@buaa.edu.cn). 

Our code is heavily borrowed from ReActNet (https://github.com/liuzechun/ReActNet).
## Dependencies
* Python 3.8
* Pytorch 1.7.1

## RBONN with two-stage tranining

We test our RBONN using the same ResNet-18 structure and training setttings as [ReActNet](https://github.com/liuzechun/ReActNet), and obtain 66.7% top-1 accuracy.

| Methods | Top-1 acc | Quantized model link |
|:-------:|:---------:|:--------------------:|
|ReActNet |  65.9     | [Model](https://github.com/liuzechun/ReActNet#models) |
| ReCU    |  66.4     | [Model](https://drive.google.com/drive/folders/1vukw5yU0gLQlERmI9_dE4R4V1eg59mEI?usp=sharing)        |
| RBONN    |  66.7     | [Model]()        |


To verify the performance of our quantized models with ReActNet-like structure on ImageNet, please use the following command:
