import torch.nn as nn
import torch.optim as optim
from models.cnn import CNN
from models.resnet import ResNet
from models.inception import Inception
from models.densenet import DenseNet
from models.efficientnet import EfficientNet
from models.mobilenet import MobileNet
from models.vit import ViT
from utils import utils

def get_criterion():
    # Define the criterion
    criterion = nn.CrossEntropyLoss()
    return criterion

def get_optimizer(model):
    # Define the optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    return optimizer

def main():
    """
    Main function to load the CIFAR10 dataset, normalize it, create data loaders for training and testing,
    train and evaluate all the models, and save the results.
    """
    # Load and normalize the CIFAR10 dataset and create data loaders
    trainloader, testloader, classes = utils.load_data()
        
    # List of models to train and evaluate
    model_classes = [CNN, ResNet, Inception, DenseNet, EfficientNet, MobileNet, ViT]
    models = [model_class() for model_class in model_classes]
    
    criterion = get_criterion()
    
    # Train and evaluate each model
    for model in models:
        optimizer = get_optimizer(model)
        print(f"Training and evaluating {model.__class__.__name__}...")
        model.train(trainloader,criterion,optimizer)
        # accuracy = model.evaluate(testloader)
        # utils.save_results(model.__class__.__name__, accuracy)
        print(f"Done with {model.__class__.__name__}.\n")

# ... rest of your code ...

if __name__ == "__main__":
    """
    Entry point of the script. Calls the main function.
    """
    main()