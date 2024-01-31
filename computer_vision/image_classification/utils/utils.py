import torch
from torchvision import datasets, transforms
import os

def load_data():
    """
    Load and normalize the CIFAR10 training and test datasets using torchvision.

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
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Load the CIFAR10 training data
    trainset = datasets.CIFAR10(root='./data', train=True,
                                download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)

    # Load the CIFAR10 test data
    testset = datasets.CIFAR10(root='./data', train=False,
                               download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=2)

    # Define the class labels
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader, classes


def save_results(model_name, accuracy):
    """
    Save the model name and its accuracy to a file.

    This function opens a file named 'accuracy.txt' in a 'results' folder in append mode, writes the model name and its accuracy to the file, and then closes the file. If the file or the folder doesn't exist, they will be created.

    Args:
        model_name (str): The name of the model.
        accuracy (float): The accuracy of the model.
    """
    # Ensure the 'results' directory exists
    if not os.path.exists('results'):
        os.makedirs('results')

    with open('results/accuracy.txt', 'a') as f:
        f.write(f'Model: {model_name}, Accuracy: {accuracy}\n')