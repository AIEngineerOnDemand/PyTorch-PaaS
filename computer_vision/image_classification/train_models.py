import subprocess
import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.cnn import CNN
from models.resnet import ResNet
from models.inception import Inception
from models.densenet import DenseNet
from models.efficientnet import EfficientNet
from models.mobilenet import MobileNet
from models.vit import ViT
from utils.utils import DummyDataset
import torchvision
import logging
import time

# Install dependencies from requirements.txt
subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

print(f"torchvision version: {torchvision.__version__}")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

def get_model_by_name(model_name):
    model_classes = {
        'CNN': CNN,
        'ResNet': ResNet,
        'Inception': Inception,
        'DenseNet': DenseNet,
        'EfficientNet': EfficientNet,
        'MobileNet': MobileNet,
        'ViT': ViT
    }
    if model_name in model_classes:
        return model_classes[model_name]()
    else:
        raise ValueError(f"Model {model_name} is not supported.")

def get_criterion():
    return nn.CrossEntropyLoss()

def get_optimizer(model):
    return optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

def train(args):
    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')

    model = get_model_by_name(args.model_name)
    model.to(device)
    criterion = get_criterion()
    optimizer = get_optimizer(model)

    if args.fast_local_mode:
        # Use DummyDataset for fast local training
        dataset = DummyDataset(args.model_name)
        dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
    else:
        # Placeholder for actual data loading logic for SageMaker training
        dataset = DummyDataset()  # Replace with actual dataset loading
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        logging.info(f'Epoch [{epoch+1}/{args.epochs}], Loss: {running_loss/len(dataloader)}')

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train a model.')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model to train')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--fast_local_mode', action='store_true', help='Use fast local mode with dummy dataset')
    args = parser.parse_args()

    train(args)

if __name__ == '__main__':
    main()