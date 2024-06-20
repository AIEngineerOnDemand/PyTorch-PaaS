from torchvision.models import densenet121
import torch.nn as nn
from .base_model import BaseModel

class DenseNet(BaseModel):
    def __init__(self, num_classes=10):
        super(DenseNet, self).__init__(num_classes)
        self.model = densenet121(pretrained=True)
        num_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(num_features, self.num_classes)
        
    def set_model(self, model):
        self.model = model