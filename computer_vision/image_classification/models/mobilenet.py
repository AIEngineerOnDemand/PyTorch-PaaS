import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from .base_model import BaseModel

class MobileNet(BaseModel):
    def __init__(self, num_classes=10):
        super(MobileNet, self).__init__(num_classes)
        # Use the updated API with weights
        weights = MobileNet_V2_Weights.IMAGENET1K_V1  # or use MobileNet_V2_Weights.DEFAULT for the most up-to-date weights
        self.model = mobilenet_v2(weights=weights)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x