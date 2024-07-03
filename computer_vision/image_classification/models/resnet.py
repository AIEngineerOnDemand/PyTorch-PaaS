import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from .base_model import BaseModel

class ResNet(BaseModel):
    def __init__(self, num_classes=10):
        super(ResNet, self).__init__(num_classes)
        # Use the updated API with weights
        weights = ResNet50_Weights.IMAGENET1K_V1  # or use ResNet50_Weights.DEFAULT for the most up-to-date weights
        self.model = resnet50(weights=weights)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x