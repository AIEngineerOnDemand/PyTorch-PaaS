import torch.nn as nn
import torch.optim as optim
from models.cnn import CNN
from models.resnet import ResNet
from models.inception import Inception
from models.densenet import DenseNet
from models.efficientnet import EfficientNet
from models.mobilenet import MobileNet
from models.vit import ViT
from utils.utils import load_data, save_model
import logging
import torch
from torch.profiler import profile, ProfilerActivity, record_function
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

def quick_test(trainloader, model):
    try:
        criterion = get_criterion()
        optimizer = get_optimizer(model)
        test_loader = torch.utils.data.DataLoader(trainloader.dataset, batch_size=2, shuffle=False, num_workers=0, 
                                                  sampler=torch.utils.data.SubsetRandomSampler(range(10)))
        with profile(activities=[ProfilerActivity.CPU], record_shapes=False, with_stack=False) as prof:
            with record_function("model_training_quick_test"):
                for i, data in enumerate(test_loader):
                    if i >= 1:  # Process only one batch to check for errors
                        break
                    images, labels = data
                    optimizer.zero_grad()
        # Log the profiling information
        logging.info(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        print(f"\nModel {model.__class__.__name__} can be trained.\n")
    except Exception as e:
        logging.error("Error during training: ", exc_info=True)
        print(f"Error during training: {e}")
    
def main(fast_local=False):
    """
    Main function to load the CIFAR10 dataset, normalize it, create data loaders for training and testing,
    train and evaluate all the models, and save the results.
    """
    # List of model names to train and evaluate
    model_names = ['CNN', 'ResNet', 'Inception', 'DenseNet', 'EfficientNet', 'MobileNet', 'ViT']
    
    for model_name in model_names:
        # Initialize the model based on model_name
        model = get_model_by_name(model_name)  # Assume get_model_by_name function exists
        
        if fast_local:
            trainloader, testloader, classes = load_data(model_name, subsample=True, subsample_rate=0.01)
            quick_test(trainloader, model)  # Pass the model instance instead of model_name
        else:
            trainloader, testloader, classes = load_data(model_name)
            
            criterion = get_criterion()
            optimizer = get_optimizer(model)
            
            print(f"Training and evaluating {model_name}...")
            model.train_model(trainloader, criterion, optimizer)
            save_model(model, f"Computer-Vision-Models/Image-Classification/{model_name}.pth")
            print(f"Done with {model_name}.\n")


if __name__ == "__main__":
    """
    Entry point of the script. Calls the main function.
    """
    main(fast_local=False)
    
    