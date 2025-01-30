# Adaptable Usage Control for Evolving AI Models.

This repository contains a Python script for training neural networks with a focus on selected partial neurons. 

[Paper](https://exapme) | [Dataset](https://pytorch.org/vision/stable/datasets.html)

## Abstract

<details><summary>Abstract</summary>


This repository contains a Python script for training neural networks with a focus on selected partial neurons. 

</details>

## Features

- **Multiple Datasets**: Supports training on popular datasets such as CIFAR-10, CIFAR-100, MNIST, FashionMNIST, Flower102, and Caltech256.
- **Model Architectures**: Includes support for ResNet-18, ResNet-152, DenseNet121, and ConvNeXtV2.
- **Neuron Selection**: Allows selection of neurons based on different criteria (Top, Random, Bottom) and retains a specified fraction of neurons.

### Environment Setup

1. Clone the repository:

```
git clone https://github.com/yaojin17/Unlearning_LLM.git
cd vision_task
```

2. Install the required packages:

```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -e .
pip install -r requirements.txt
```

*All experiments are conducted on NVIDIA RTX4090 GPU with 24GB of memory.*

#### Usage

To train a model, run the following command with the desired arguments:

```
python main.py --dataset cifar100 --model_name densenet121
```

#### Arguments

- `--device`: Training device (`cuda:0` or `cpu`). Default is `cuda:0` if available, otherwise `cpu`.
- `--dataset`: Dataset for training. Choices: `cifar100`, `cifar10`, `mnist`, `fashionmnist`, `flower102`, `caltech256`. Default is `cifar100`.
- `--batch_size`: Default is `64`.
- `--num_workers`: Default is `6`.
- `--num_epochs`: Number of training epochs. Default is `50`.
- `--lr`: Learning rate. Default is `0.001`.
- `--momentum`: Momentum for SGD optimizer. Default is `0.9`.
- `--model_name`: Model architecture to use. Choices: `resnet18`, `resnet152`, `densenet121`, `convnextv2`. Default is `densenet121`.
- `--pretrain`: Use pretrained model weights. Default is `True`.
- `--num_classes`: Number of output classes. Default is `100` (cifar100).
- `--retained`: Fraction of neurons to retain. Default is `0.05`.
- `--mode`: Neuron selection mode. `0` for Top, `1` for Random, `2` for Bottom. Default is `0`.

***Note***: For Caltech256 dataset you need to download from [Kaggle](https://www.kaggle.com/datasets/jessicali9530/caltech256).

### Example

Train a DenseNet121 model on the Caltech256 dataset with pretrained weights, retaining the top 5% of neurons:

```
python main.py --dataset caltech256 --model_name densenet121 --pretrain --retained 0.05 --mode 0 --batch_size 64 --num_epochs 50 --lr 0.001 --momentum 0.9
```

## Citation Information

If you find this code or dataset useful, please consider citing our paper:

```bib
@{comming soon}
```

