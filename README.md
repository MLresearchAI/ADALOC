# Adaptable Usage Control for Evolving AI Models.

This repository contains a Python script for training neural networks with a focus on selected partial neurons. 

[Paper](https://exapme) | [Dataset](https://pytorch.org/vision/stable/datasets.html)

## Abstract

<details><summary>Abstract</summary>


This repository provides a unified framework for training and evaluating models on both vision and natural language processing (NLP) tasks. The project supports multiple datasets and model architectures across two domains, with flexible configuration options for researchers.

</details>

## Features

**Vision Tasks**

- Supports training on popular datasets such as CIFAR-10, CIFAR-100, MNIST, FashionMNIST, Flower102, and Caltech256.

- Includes support for ResNet-18, ResNet-152, DenseNet121, and ConvNeXtV2.
- Allows selection of neurons based on different criteria (Top, Random, Bottom) and retains a specified fraction of neurons.

**NLP Tasks**

- Supports multiple datasets: `qnli`, `sst2`, `tweet_eval`
- Adjustable parameter selection for different percentages of data
- Training and evaluation with different models such as BERT, RoBERTa, and DeBERTa
- Automatic logging and result storage

## Vision task

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
python tune_vision.py --dataset cifar100 --model_name densenet121
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
python tune_vision.py --dataset caltech256 --model_name densenet121 --pretrain --retained 0.05 --mode 0 --batch_size 64 --num_epochs 50 --lr 0.001 --momentum 0.9
```

## NLP task

### Requirements

1. Make sure you have the following dependencies installed:

   ```
   pip install torch transformers datasets evaluate numpy pandas
   ```

## Usage

### Running the Script

```bash
python script.py --model_name <model> --output_path <path> --mode <mode>
```

### Arguments

| Argument        | Type   | Description                                                  |
| --------------- | ------ | ------------------------------------------------------------ |
| `--model_name`  | string | Choose from `bert-base-uncased`, `roberta-base`, `microsoft/deberta-v3-base` |
| `--output_path` | string | Path to save the final results                               |
| `--mode`        | string | Choose from `smallest`, `largest`, `random`                  |

### Example Command

```bash
python script.py --model_name bert-base-uncased --output_path ./results --mode largest
```

## Script Overview

### Dataset Loading

- The script loads datasets based on the given dataset name.
- Supported datasets: `qnli`, `sst2`, `tweet_eval`, `rotten_tomatoes`, `yahoo_answers_topics`

### Data Processing

- Tokenization is performed based on dataset content.
- Selection of a subset of data for training and evaluation.

### Model Training

- Pre-trained models are initialized and fine-tuned.
- A threshold is set to select top parameters based on absolute values.

### Evaluation

- Models are evaluated using predefined metrics such as accuracy.
- Results are stored in a CSV file.

## Output

After running the script, the following outputs will be generated:

1. A CSV file containing evaluation results (`final_results.csv`).
2. Logging directory with performance logs.
3. Printed evaluation metrics such as accuracy and loss.

## Example Output Format

The resulting CSV file (`final_results.csv`) contains:

| dataset | p    | accuracy | loss |
| ------- | ---- | -------- | ---- |
| qnli    | 0.05 | -        | -    |
| sst2    | 0.1  | -        | -    |





Citation Information

If you find this code or dataset useful, please consider citing our paper:

```bib
@{comming soon}
```

