import argparse
import torch


from data import dataset_loader
from helper import load_pretrained_model, neurons_selection, train_model, set_seed


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu",
                        help="Training device: 'cuda:0' or 'cpu'")

    # dataset
    parser.add_argument("--dataset", type=str, default="cifar100",
                        choices=["cifar100", "cifar10", "mnist", "fashionmnist", "flower102", "caltech256"],
                        help="Dataset for training")

    # hyperparameters
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--num_workers", type=int, default=6, help="Number of workers for data loading")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD optimizer")

    # model config
    parser.add_argument("--model_name", type=str, default="densenet121",
                        choices=["resnet18","resnet152", "densenet121", "convnextv2"],
                        help="Model architecture to use")
    parser.add_argument("--pretrain", action="store_true", help="Use pretrained model weights")
    parser.add_argument("--num_classes", type=int, default=100, help="Number of output classes")

    # neuron selection
    parser.add_argument("--retained", type=float, default=0.05, help="Fraction of neurons to retain")
    parser.add_argument("--mode", type=int, choices=[0, 1, 2], default=0, help="0 for Top, 1 for random, 2 for bottom")

    return parser.parse_args()

def main():
    args = parse_args()
    print(f"Configuration: {args}")
    set_seed(42)
    device = torch.device(args.device)
    train_loader, valid_loader = dataset_loader(dataset_name=args.dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    model = load_pretrained_model(args.model_name, num_classes=args.num_classes, pretrain=args.pretrain)
    selected_neurons = neurons_selection(model, retained=args.retained, mode=args.mode)
    train_model(model, args.model_name, train_loader, valid_loader, device, selected_neurons,
                num_epochs=args.num_epochs, lr=args.lr, momentum=args.momentum)

if __name__ == "__main__":
    main()
