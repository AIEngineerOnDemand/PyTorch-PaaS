import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from models.cnn import CNN
from models.resnet import ResNet
from models.inception import Inception
from models.densenet import DenseNet
from models.efficientnet import EfficientNet
from models.mobilenet import MobileNet
from models.vit import ViT
from utils.utils import load_data, save_model,get_transform_for_model
import logging
import torch
from PIL import Image
from torch.profiler import profile, ProfilerActivity, record_function
import argparse
# Configure logging
logging.basicConfig(filename='training_log.log', level=logging.INFO, 
                    format='%(asctime)s:%(levelname)s:%(message)s')

def get_model_by_name(model_name):
    """
    Returns an instance of the model class based on the model_name argument.
    """
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
    # Define the criterion
    criterion = nn.CrossEntropyLoss()
    return criterion

def get_optimizer(model):
    # Define the optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    return optimizer

class DummyDataset(Dataset):
    def __init__(self, model_name, num_samples=100, num_classes=10):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.transform = get_transform_for_model(model_name)
        # Generate dummy data and labels
        if model_name == 'Inception':
            self.data = torch.randn(num_samples, 3, 299, 299)
        else:
            self.data = torch.randn(num_samples, 3, 224, 224)
        self.targets = torch.randint(0, num_classes, (num_samples,))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Convert tensor to PIL Image
        img = transforms.ToPILImage()(self.data[idx])
        target = self.targets[idx]
        # Apply transformations
        img = self.transform(img)
        return img, target
    
def get_dummy_dataloader(model_name, batch_size=10, num_samples=100, num_classes=10):
    dataset = DummyDataset(model_name, num_samples=num_samples, num_classes=num_classes)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def test_train_model(model_name, local_fast=False):
    model = get_model_by_name(model_name)
    
    if local_fast:
        trainloader = get_dummy_dataloader(model_name)
    else:
        # Replace with your actual data loading logic
        trainloader, _, classes = load_data(model_name)
               
    criterion = get_criterion()
    optimizer = get_optimizer(model)
    model.train_model(trainloader, criterion, optimizer)
    if not local_fast:
       save_model(model, f"Computer-Vision-Models/Image-Classification/{model_name}.pth")
    print(f" The model - {model_name} - has been trained.\n")

def main(fast_local=False):
    """
    Main function to load the CIFAR10 dataset, normalize it, create data loaders for training and testing,
    train and evaluate all the models, and save the results.
    """
    # List of model names to train and evaluate
    model_names = ['CNN', 'ResNet', 'Inception', 'DenseNet', 'EfficientNet', 'MobileNet', 'ViT']

    for model_name in model_names:
        # Initialize the model based on model_name
        #with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        test_train_model(model_name, fast_local)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train models with optional fast local mode.')
    parser.add_argument('--fast_local', action='store_true', help='Enable fast local mode for training')
    args = parser.parse_args()
    main(fast_local=args.fast_local)

    