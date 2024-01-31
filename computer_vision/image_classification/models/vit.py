import torch.nn as nn
import timm
from .base_model import BaseModel

class ViT(BaseModel):
    def __init__(self, num_classes=10):
        super(ViT, self).__init__(num_classes)
        self.model = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.model.head = nn.Linear(self.model.head.in_features, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x