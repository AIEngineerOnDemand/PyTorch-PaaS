import torch
from torchvision import datasets, transforms
import os
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Subset, DataLoader,Dataset 
import numpy as np
import boto3
import logging


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
        # Convert tensor to PIL Image, apply transformations, and return
        img = transforms.ToPILImage()(self.data[idx])
        target = self.targets[idx]
        img = self.transform(img)
        return img, target

def get_dummy_dataloader(model_name, batch_size=10, num_samples=100, num_classes=10):
    dataset = DummyDataset(model_name, num_samples=num_samples, num_classes=num_classes)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def get_transform_for_model(model_name):
    if model_name == 'Inception':
        resize_transform = transforms.Resize((299, 299))
    elif model_name in ['CNN', 'ResNet', 'DenseNet', 'EfficientNet', 'MobileNet', 'ViT']:
        resize_transform = transforms.Resize((224, 224))
    else:
        # Default to 224x224 if model name is unknown
        resize_transform = transforms.Resize((224, 224))
    
    return transforms.Compose([
        resize_transform,
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

def load_data(model_name,subsample=False, subsample_rate=False):
    """

    This function performs the following steps:

    1. Defines a transform that will be applied to the images in the dataset. This transform first converts the images to PyTorch tensors (with transforms.ToTensor()) and then normalizes the tensors with mean and standard deviation of 0.5 for each of the three color channels (with transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))).

    2. Loads the CIFAR10 training and test datasets, applies the transform to them, and wraps them in a DataLoader. The DataLoader will provide batches of 4 images at a time, shuffling the training data but not the test data. It will use 2 worker processes to load the data in parallel.

    3. Defines the class labels for the CIFAR10 dataset.

    The function returns the training and test data loaders and the class labels. The returned trainloader and testloader objects can be iterated over to get batches of images and their corresponding labels. The classes list can be used to map the numeric labels to their corresponding class names.

    Note: The batch size and number of worker processes are hard-coded. Depending on your specific use case and the resources available on your machine, you might want to make these parameters configurable.

    Returns:
        trainloader (torch.utils.data.DataLoader): The DataLoader for the training data.
        testloader (torch.utils.data.DataLoader): The DataLoader for the test data.
        classes (tuple of str): The class labels for the CIFAR10 dataset.
    """
    # Define a transform to normalize the data
    """
    This code defines a transformation pipeline for preprocessing images.
    
    The transformation pipeline is composed of two steps:
    1. `transforms.ToTensor()`: This converts the input PIL Image to a PyTorch tensor. It also scales the image's pixel intensity values in the range 0-255 to a float in the range 0.0-1.0.
    2. `transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))`: This normalizes the tensor image with mean and standard deviation. 
        The mean and standard deviation are specified for all three channels (R, G, B), and they are both (0.5, 0.5, 0.5). This will scale the pixel values from the range 0.0-1.0 to the range -1.0-1.0.
    
    """

    """
    Loads the CIFAR10 test dataset.

    This function uses the torchvision.datasets.CIFAR10 class to load the CIFAR10 test dataset. The dataset is downloaded if it is not already present in the './data' directory. The images in the dataset are transformed using the transform defined earlier in the code.

    Args:
        root (str): The path to the directory where the dataset will be stored. In this case, it's './data'.
        train (bool): If False, creates a dataset from the test set. In this case, it's False.
        download (bool): If True, downloads the dataset from the internet if it's not available at root. In this case, it's True.
        transform (callable, optional): A function/transform that takes in an PIL image and returns a transformed version. In this case, it's the transform defined earlier in the code.

    Returns:
        testset (torchvision.datasets.CIFAR10): The CIFAR10 test dataset loaded and transformed.
    """
    transform = get_transform_for_model(model_name)
    
    # Load the full CIFAR10 datasets
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    if subsample:
        # Subsample the datasets
        train_indices = np.random.choice(len(trainset), int(len(trainset) * subsample_rate), replace=False)
        test_indices = np.random.choice(len(testset), int(len(testset) * subsample_rate), replace=False)

        trainset = Subset(trainset, train_indices)
        testset = Subset(testset, test_indices)

    trainloader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

    return trainloader, testloader


def save_model(model, s3_path):
    # Save the model locally
    local_path = 'model.pth'
    torch.save(model.state_dict(), local_path)
    logging.info(f"Model saved locally at {local_path}")

    # Parse the S3 path
    s3_parts = s3_path.replace("s3://", "").split("/")
    bucket_name = s3_parts[0]
    s3_key = "/".join(s3_parts[1:])
    logging.info(f"Parsed S3 path: bucket_name={bucket_name}, s3_key={s3_key}")

    # Upload the model to S3
    s3_client = boto3.client('s3')
    try:
        s3_client.upload_file(local_path, bucket_name, s3_key)
        logging.info(f"Model uploaded to S3 at s3://{bucket_name}/{s3_key}")
    except Exception as e:
        logging.error(f"Failed to upload model to S3: {e}")

    # Remove the local model file
    try:
        os.remove(local_path)
        logging.info(f"Local model file {local_path} removed")
    except Exception as e:
        logging.error(f"Failed to remove local model file: {e}")

def load_model(filename, model_class):
    """
    Load the model parameters.

    Args:
        filename (str): The path from where to load the model.
        model_class (class): The class of the model.

    Returns:
        model (torch.nn.Module): The PyTorch model loaded.
    """
    model = model_class()
    model.load_state_dict(torch.load(filename))
    return model
def save_results(model_name, accuracy, precision, recall, f1, confusion, auc_roc, logloss):
    """
    Save the model name and its metrics to a file.

    This function opens a file named 'results.md' in a 'results' folder in append mode, writes the model name and its metrics to the file in a table format, and then closes the file. If the file or the folder doesn't exist, they will be created.

    Args:
        model_name (str): The name of the model.
        accuracy (float): The accuracy of the model.
        precision (float): The precision of the model.
        recall (float): The recall of the model.
        f1 (float): The F1 score of the model.
        confusion (np.array): The confusion matrix of the model.
        auc_roc (float): The AUC-ROC of the model.
        logloss (float): The log loss of the model.
    """
    # Ensure the 'results' directory exists
    if not os.path.exists('results'):
        os.makedirs('results')

    with open('results/results.md', 'a') as f:
        f.write(f'| Model | Accuracy | Precision | Recall | F1 Score | AUC-ROC | Log Loss |\n')
        f.write(f'| --- | --- | --- | --- | --- | --- | --- |\n')
        f.write(f'| {model_name} | {accuracy:.2f} | {precision:.2f} | {recall:.2f} | {f1:.2f} | {auc_roc:.2f} | {logloss:.2f} |\n')
        f.write(f'\n![Confusion Matrix for {model_name}](https://github.com/AIEngineerOnDemand/PyTorch-Use-Cases/blob/master/computer_vision/image_classification/results/{model_name}_confusion_matrix.png)\n')

    # Plot confusion matrix
    plt.figure(figsize=(10,7))
    sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f'results/{model_name}_confusion_matrix.png')