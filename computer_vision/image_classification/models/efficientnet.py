import torch.nn as nn
import timm
from .base_model import BaseModel

class EfficientNet(BaseModel):
    def __init__(self, num_classes=10):
        super(EfficientNet, self).__init__(num_classes)
        self.model = timm.create_model('efficientnet_b0', pretrained=True)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x