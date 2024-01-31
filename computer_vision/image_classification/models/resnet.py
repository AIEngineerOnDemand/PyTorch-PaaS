import torch.nn as nn
from torchvision.models import resnet50
from .base_model import BaseModel

class ResNet(BaseModel):
    def __init__(self, num_classes=10):
        super(ResNet, self).__init__(num_classes)
        self.model = resnet50(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x