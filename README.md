# AdaLoc

AdaLoc enables secure, adaptable DNN usage control through a localized update using a small, adaptable key. It allows efficient updates to a small subset of parameters while maintaining strong performance and protecting against unauthorized access.

## Features

- **Vision Tasks**: Supports datasets like CIFAR-10, CIFAR-100, MNIST, FashionMNIST, Flower102, and Caltech256. Includes models such as ResNet-18, ResNet-152, DenseNet121, and ConvNeXtV2. Allows neuron selection based on criteria like Top, Random, or Bottom.
- **NLP Tasks**: Supports datasets such as QNLI, SST-2, and TweetEval. Compatible with BERT, RoBERTa, and DeBERTa models, with automatic logging and result storage.

## Vision Task

### Environment Setup

1. Clone the repository:

   ```
   git clone https://github.com/MLresearchAI
   ```

2. Install required packages:

   ```
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install -e .
   pip install -r requirements.txt
   ```

### Usage

To train a model, run the following:

```
python tune_v.py --dataset cifar100 --model_name densenet121
```

### Arguments

- `--device`: Training device (`cuda:0` or `cpu`).
- `--dataset`: Dataset for training (`cifar100`, `cifar10`, `mnist`, `fashionmnist`, `flower102`, `caltech256`).
- `--batch_size`: Default `64`.
- `--num_workers`: Default `6`.
- `--num_epochs`: Default `50`.
- `--lr`: Default `0.001`.
- `--momentum`: Default `0.9`.
- `--model_name`: Model architecture (`resnet18`, `resnet152`, `densenet121`, `convnextv2`).
- `--pretrain`: Use pretrained model weights (`True` by default).
- `--num_classes`: Number of output classes (`100` by default).
- `--retained`: Fraction of neurons to retain (`0.05` by default).
- `--mode`: Neuron selection mode (`0` for Top, `1` for Random, `2` for Bottom).

**Note**: For the Caltech256 dataset, download it from [Kaggle](https://www.kaggle.com/datasets/jessicali9530/caltech256).

### Example

To train a DenseNet121 model on the Caltech256 dataset:

```
python tune_v.py --dataset caltech256 --model_name densenet121 --pretrain --retained 0.05 --mode 0 --batch_size 64 --num_epochs 50 --lr 0.001 --momentum 0.9
```

## NLP Task

### Requirements

Ensure the following dependencies are installed:

```
pip install torch transformers datasets evaluate numpy pandas
```

### Usage

Run the script as follows:

```bash
python tune_l.py --model_name <model> --output_path <path> --mode <mode>
```

### Arguments

| Argument        | Type   | Description                                                  |
| --------------- | ------ | ------------------------------------------------------------ |
| `--model_name`  | string | Choose from `bert-base-uncased`, `roberta-base`, `microsoft/deberta-v3-base` |
| `--output_path` | string | Path to save the final results                               |
| `--mode`        | string | Choose from `smallest`, `largest`, `random`                  |

### Example Command

```bash
python tune_l.py --model_name bert-base-uncased --output_path ./results --mode largest
```

## Output

After running the script, the following outputs are generated:

1. A CSV file containing evaluation results (`final_results.csv`).
2. A logging directory with performance logs.
3. Printed evaluation metrics like accuracy and loss.

### Example Output Format

The resulting CSV file (`final_results.csv`) contains:

| dataset | p    | accuracy | loss |
| ------- | ---- | -------- | ---- |
| qnli    | 0.05 | -        | -    |
| sst2    | 0.1  | -        | -    |

---

## Model Keying 

Our adaptable key is designed to support and integrate seamlessly with a wide range of model keying methods. A demonstration implementation is available in the `model_keying/` directory.

```python
cd model_keying/

# To split the target model (primary key):
python locker.py split --model <path_to_model> --ratio <key_ratio> --arch <architecture> --save ./test_output

# To restore the model:
python locker.py recover --model test_output/keyed_model.pth.tar --key test_output/key.pth.tar --arch <architecture> --save <save_path>
```



## Citation Information

If you find this code or dataset useful, please consider citing our paper:

```bib
@{coming soon}
```

