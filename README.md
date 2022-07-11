# Recurrent Bilinear Optimization for Binary Neural Networks (RBONN)
Pytorch implementation of RBONN in ECCV 2022.
## Tips

Any problem, please contact the first author (Email: shengxu@buaa.edu.cn). 

## Dependencies
* Python 3.8
* Pytorch 1.7.1

## Comparison with SOTAs

We test our ReCU using the same ResNet-18 structure and training setttings as [ReActNet](https://github.com/liuzechun/ReActNet), and obtain higher top-1 accuracy.

| Methods | Top-1 acc | Quantized model link |
|:-------:|:---------:|:--------------------:|
|ReActNet |  65.9     | [ReActNet (Bi-Real based)](https://github.com/liuzechun/ReActNet#models) |
| ReCU    |  66.4     | [ResNet-18](https://drive.google.com/drive/folders/1vukw5yU0gLQlERmI9_dE4R4V1eg59mEI?usp=sharing)        |


To verify the performance of our quantized models with ReActNet-like structure on ImageNet, please use the following command:
```bash
cd imagenet_two-stage && python -u evaluate.py \
python -u main.py \
--gpus 0 \
-e [best_model_path] \
--model resnet18_1w1a \
--data_path [DATA_PATH] \
--dataset imagenet \
-bt 256 \
```
