# PyTorch Image Classification Models

In this project, we explore several popular deep learning models used for image classification tasks with PyTorch. Here are the models we will be experimenting with:

- **Convolutional Neural Networks (CNNs)**: These are the most commonly used models for image classification tasks. They consist of convolutional layers, pooling layers, and fully connected layers. LeNet, AlexNet, and VGGNet are examples of CNNs.

- **ResNet (Residual Network)**: This is a type of CNN that introduces "skip connections" or "shortcuts" to allow the gradient to be directly backpropagated to earlier layers. It's known for its ability to train very deep networks.

- **Inception (GoogLeNet)**: This model introduces a new module called the "Inception module" that allows for more efficient computation and deeper networks.

- **DenseNet (Densely Connected Convolutional Networks)**: In this model, each layer is connected to every other layer in a feed-forward fashion, which can lead to lower complexity models.

- **EfficientNet**: This is a newer model that scales the width, depth, and resolution of the network for better performance.

- **MobileNet**: This model is designed for mobile and embedded vision applications. It uses depthwise separable convolutions to reduce the model size and complexity, making it suitable for mobile applications.

- **Transformer Models (like Vision Transformer - ViT)**: While originally designed for NLP tasks, Transformer models have recently been adapted for image classification tasks and have shown competitive results.

We will be training and evaluating all these models on our dataset and comparing their performance. The choice of model depends on the specific task, the size and nature of our dataset, computational resources, and the trade-off between speed and accuracy that we're willing to make.

PyTorch's flexibility and efficient tensor computations make it suitable for tasks like image classification, object detection, and image generation.

When evaluating image classification models in PyTorch, several common metrics are used:

- **Accuracy**: The ratio of correctly predicted observations to the total observations.
- **Precision**: The ratio of correctly predicted positive observations to the total predicted positives.
- **Recall (Sensitivity)**: The ratio of correctly predicted positive observations to all actual positives.
- **F1 Score**: The weighted average of Precision and Recall.
- **Confusion Matrix**: A table that describes the performance of a classification model on a set of test data for which the true values are known.
- **Area Under the ROC Curve (AUC-ROC)**: A plot of the true positive rate against the false positive rate.
- **Log Loss**: Often used in place of accuracy.

## Running the Training on SageMaker

This project is designed to run on AWS SageMaker using either local mode or AWS training mode. We use the SageMaker SDK and CLI to set up and execute the training jobs. Below is a brief overview of the key scripts involved:

### `run_training.py`

This script initializes the SageMaker session, creates a temporary directory for dummy data, dynamically constructs the `source_dir` path, and creates a PyTorch estimator. The estimator is then fitted using either local mode or AWS training mode based on the `execution_mode` parameter.

### `train_models.py`

This script installs dependencies, configures logging, and defines the training process. It includes functions to get the model by name, criterion, and optimizer. The main training loop handles the training process, including loading the dataset, moving data to the appropriate device (CPU/GPU), and logging the training progress.

### Example Usage

To run the training script, execute the following command:

```sh
python run_training.py --execution_mode fast_local_mode
```