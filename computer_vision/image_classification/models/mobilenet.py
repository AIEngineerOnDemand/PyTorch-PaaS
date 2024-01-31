import torch.nn as nn
from torchvision.models import mobilenet_v2
from .base_model import BaseModel

class MobileNet(BaseModel):
    def __init__(self, num_classes=10):
        super(MobileNet, self).__init__(num_classes)
        self.model = mobilenet_v2(pretrained=True)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x