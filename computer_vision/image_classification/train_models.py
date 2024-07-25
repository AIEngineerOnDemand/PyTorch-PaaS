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
import logging
import time

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
    model = get_model_by_name(args.model_name)
    criterion = get_criterion()
    optimizer = get_optimizer(model)

    if args.fast_local_mode:
        # Use DummyDataset for fast local training
        dataset = DummyDataset(args.model_name)
        dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
    else:
        # Placeholder for actual data loading logic for SageMaker training
        pass

    # Training loop
    model.train_model(dataloader, criterion, optimizer, args.epochs)

    # Save the trained model with a unique name
    model_dir = args.model_dir
    model_name = args.model_name
    timestamp = int(time.time())  # Use timestamp as a simple way to generate a unique identifier

    model_filename = f'{model_name}_{timestamp}_model.pth'
    model_info_filename = f'{model_name}_{timestamp}_model_info.pth'

    with open(os.path.join(model_dir, model_filename), 'wb') as f:
        torch.save(model.state_dict(), f)

    with open(os.path.join(model_dir, model_info_filename), 'wb') as f:
        torch.save({'model_name': model_name}, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # SageMaker container environment
    parser.add_argument('--model_name', type=str, default='MobileNet')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', '.'))
    parser.add_argument('--fast_local_mode', action='store_true', help='Use DummyDataset for fast local training, some models take too long to train locally even with a small sample of the dataset')

    args = parser.parse_args()

    train(args)