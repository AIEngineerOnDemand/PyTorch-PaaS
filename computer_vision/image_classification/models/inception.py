import torch.nn as nn
from torchvision import models
from .base_model import BaseModel

class Inception(BaseModel):
    def __init__(self, num_classes=10):
        super(Inception, self).__init__(num_classes)
        self.model = models.inception_v3(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x