import torch
from models import cnn, resnet, inception, densenet, efficientnet, mobilenet, vit
from utils import utils

def main():
    """
    Main function to load the CIFAR10 dataset, normalize it, create data loaders for training and testing,
    train and evaluate all the models, and save the results.
    """
    # Load and normalize the CIFAR10 dataset and create data loaders
    trainloader, testloader, classes = utils.load_data()

    # List of models to train and evaluate
    models = [cnn, resnet, inception, densenet, efficientnet, mobilenet, vit]

    # Train and evaluate each model
    for model in models:
        print(f"Training and evaluating {model.__name__}...")
        model.train(trainloader)
        accuracy = model.evaluate(testloader)
        utils.save_results(model.__name__, accuracy)
        print(f"Done with {model.__name__}.\n")

if __name__ == "__main__":
    """
    Entry point of the script. Calls the main function.
    """
    main()