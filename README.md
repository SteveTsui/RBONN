# Recurrent Bilinear Optimization for Binary Neural Networks (RBONN)
Pytorch implementation of our RBONN accepted by ECCV2022 as oral presentation.
## Tips

Any problem, please contact the first author (Email: shengxu@buaa.edu.cn). 

Our code is heavily borrowed from ReActNet (https://github.com/liuzechun/ReActNet).
## Dependencies
* Python 3.8
* Pytorch 1.7.1
* Torchvision 0.8.2

## RBONN with two-stage tranining

We test our RBONN using the same ResNet-18 structure and training setttings as [ReActNet](https://github.com/liuzechun/ReActNet), and obtain 66.7% top-1 accuracy.

| Methods | Top-1 acc | Top-5 acc | Quantized model link |Log|
|:-------:|:---------:|:---------:|:--------------------:|:---:|
|[ReActNet](https://arxiv.org/abs/2003.03488) |  65.9     |  -     | [Model](https://github.com/liuzechun/ReActNet#models) |-|
| [ReCU](https://arxiv.org/abs/2103.12369)    |  66.4     |  86.5     | [Model](https://github.com/z-hXu/ReCU)        |-|
| RBONN    |  66.7     |  87.0     | [Model](https://drive.google.com/drive/folders/1ZHRLyQ4ZkrhCPT2fITKq47ZLwSlMZWFx?usp=sharing)        |[Log](https://drive.google.com/drive/folders/1ZHRLyQ4ZkrhCPT2fITKq47ZLwSlMZWFx?usp=sharing)|


To verify the performance of our quantized models with ReActNet-like structure on ImageNet, please do as the following steps:
1. Finish the first stage training using [ReActNet](https://github.com/liuzechun/ReActNet).
2. Use the following command:
```bash 
cd 2_step2_rbonn
bash run.sh
```

Other models will be open-sourced successively.
