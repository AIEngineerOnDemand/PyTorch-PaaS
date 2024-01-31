import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_model import BaseModel

class CNN(BaseModel):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__(num_classes)
        self.conv1 = nn.Conv2d(3, 6, 5)  # 3 input channels, 6 output channels, 5x5 kernel
        self.pool = nn.MaxPool2d(2, 2)  # 2x2 max pooling
        self.conv2 = nn.Conv2d(6, 16, 5)  # 6 input channels, 16 output channels, 5x5 kernel
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # Fully connected layer (16*5*5 inputs, 120 outputs)
        self.fc2 = nn.Linear(120, 84)  # Fully connected layer (120 inputs, 84 outputs)
        self.fc3 = nn.Linear(84, num_classes)  # Fully connected layer (84 inputs, num_classes outputs)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # First conv layer -> ReLU -> Pooling
        x = self.pool(F.relu(self.conv2(x)))  # Second conv layer -> ReLU -> Pooling
        x = x.view(-1, 16 * 5 * 5)  # Flatten the tensor
        x = F.relu(self.fc1(x))  # First fully connected layer -> ReLU
        x = F.relu(self.fc2(x))  # Second fully connected layer -> ReLU
        x = self.fc3(x)  # Third fully connected layer
        return x
    
   