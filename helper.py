from torch import optim
import torchvision.models as models
import timm
import torch
import numpy as np
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_pretrained_model(model_name, num_classes=None, pretrain=True):
    if model_name.lower() == 'resnet152':
        model = models.resnet152(pretrained=pretrain)
    elif model_name.lower() == 'resnet18':
        model = models.resnet18(pretrained=pretrain)
    elif model_name.lower() == 'densenet121':
        model = models.densenet121(pretrained=pretrain)
    elif model_name.lower() == 'convnextv2':
        model = timm.create_model('convnextv2_tiny', pretrained=pretrain)
    else:
        raise ValueError("Unsupported model. Please check the model name and try again.")
    if num_classes == None:
        return model
    if 'resnet' in model_name.lower():
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, num_classes)
    elif 'densenet' in model_name.lower():
        num_ftrs = model.classifier.in_features
        model.classifier = torch.nn.Linear(num_ftrs, num_classes)
    elif 'convnextv2' in model_name.lower():
        num_ftrs = model.head.fc.in_features
        model.head.fc = torch.nn.Linear(num_ftrs, num_classes)
    return model


def compute_weights_norm(model):
    neuron_weights_l1_norm = []
    for name, param in model.named_parameters():
        if "weight" in name:
            for i in range(param.shape[0]):
                neuron_weights_l1_norm.append((name + "_" + str(i), torch.norm(param[i], p=1)))
    return neuron_weights_l1_norm


def neurons_selection(model, retained=1, mode=0):
    weights_norm = compute_weights_norm(model)
    sorted_weights_norm = sorted(weights_norm, key=lambda x: x[1], reverse=True)
    num_to_select = int(len(sorted_weights_norm) * retained)
    if mode == 0:
        selected_neurons = [name for name, _ in sorted_weights_norm[:num_to_select]]
    elif mode == 1:
        random_index = np.random.randint(0, len(sorted_weights_norm), num_to_select)
        selected_neurons = [sorted_weights_norm[i][0] for i in random_index]
    elif mode == 2:
        selected_neurons = [name for name, _ in sorted_weights_norm[-num_to_select:]]

    return selected_neurons


def evaluate(loader, model, criterion, device):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    average_loss = running_loss / len(loader)
    return accuracy, average_loss


def train_model(model, model_name, train_loader, valid_loader, device, selected_neurons, num_epochs=10, lr=0.001,
                momentum=0.9):
    criterion = torch.nn.CrossEntropyLoss()
    model.to(device)

    if model_name == 'convnextv2':
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    grad_masks = {}
    for name, param in model.named_parameters():
        if "weight" in name:
            mask = torch.ones_like(param, dtype=torch.float32)
            for i in range(param.shape[0]):
                if name + "_" + str(i) not in selected_neurons:
                    mask[i] = 0
            grad_masks[name] = mask

    train_accuracy, train_loss = evaluate(train_loader, model, criterion, device)
    valid_accuracy, valid_loss = evaluate(valid_loader, model, criterion, device)
    print(f'Epoch [{0}/{num_epochs}] | Loss: {train_loss:.3f} | '
          f'Train Accuracy: {train_accuracy:.2f}% | Validation Accuracy: {valid_accuracy:.2f}%')

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            for name, param in model.named_parameters():
                if param.grad is not None and name in grad_masks:
                    param.grad *= grad_masks[name]

            optimizer.step()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()
        train_accuracy = 100 * correct / total

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        valid_accuracy = 100 * correct / total
        print(f'Epoch [{epoch + 1}/{num_epochs}] | Loss: {running_loss / len(train_loader):.3f} | '
              f'Train Accuracy: {train_accuracy:.2f}% | Validation Accuracy: {valid_accuracy:.2f}%')
