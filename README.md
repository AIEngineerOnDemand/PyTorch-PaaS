# PyTorch Use Cases

This repository is dedicated to showcasing the main use cases of PyTorch, a popular open-source machine learning library. PyTorch is widely used in the tech industry for a variety of tasks, including:

## Computer Vision

PyTorch's flexibility and efficient tensor computations make it suitable for tasks like image classification, object detection, and image generation.

## Natural Language Processing (NLP)

Companies use PyTorch for tasks like text classification, sentiment analysis, machine translation, and more.

## Reinforcement Learning

PyTorch is used to develop models that can learn to make decisions from their experiences.

## Generative Models

PyTorch is used in creating generative models like GANs (Generative Adversarial Networks) which can generate new content.

## Research

Due to its dynamic computation graph and efficient memory usage, PyTorch is also a popular choice in the research community.

Stay tuned for more updates and examples of PyTorch use cases.

## Running the Code on Google Colab

Follow these steps to run the `main.py` script on Google Colab:

1. Open a new Google Colab notebook.

2. Clone the repository into the current directory in your Google Colab environment by running the following command in a new cell:

```python
!git clone https://github.com/AIEngineerOnDemand/PyTorch-Use-Cases.git
```

Running the Code on a GPU
To run your Google Colab notebook on a GPU, you can follow these steps:

Click on the 'Runtime' menu in the top toolbar.
Select 'Change runtime type' from the dropdown menu.
In the pop-up window, under 'Hardware accelerator', select 'GPU' from the dropdown menu.
Click 'Save'.
After doing this, your notebook will have access to a GPU, and any TensorFlow or PyTorch code you run will automatically use the GPU.

You can verify that your notebook is using a GPU by running the following code:

```python
import torch
print(torch.cuda.is_available())
```
This will print True if a GPU is available and False otherwise.


3.Navigate to the cloned repository:

```python
%cd PyTorch-Use-Cases/computer_vision/image_classification
```