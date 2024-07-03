from torchvision.models import densenet121, DenseNet121_Weights
import torch.nn as nn
from .base_model import BaseModel

class DenseNet(BaseModel):
    def __init__(self, num_classes=10):
        super(DenseNet, self).__init__(num_classes)
        # Use the updated API with weights
        weights = DenseNet121_Weights.IMAGENET1K_V1  # or use DenseNet121_Weights.DEFAULT for the most up-to-date weights
        self.model = densenet121(weights=weights)
        num_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(num_features, self.num_classes)
        
    def set_model(self, model):
        self.model = model