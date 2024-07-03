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
        
        # Dummy input to calculate size
        dummy_input = torch.autograd.Variable(torch.zeros(1, 3, 224, 224))
        dummy_output = self.pool(F.relu(self.conv1(dummy_input)))
        dummy_output = self.pool(F.relu(self.conv2(dummy_output)))
        self.num_flat_features = dummy_output.numel()  # Calculate total tensor size
        
        self.fc1 = nn.Linear(self.num_flat_features, 120)  # Dynamically set input size
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, self.num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.num_flat_features)  # Use dynamically calculated size
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x