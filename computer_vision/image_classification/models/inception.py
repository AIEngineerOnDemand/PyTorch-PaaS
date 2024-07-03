from .base_model import BaseModel
import torch.nn as nn
from torchvision import models
from torchvision.models.inception import Inception_V3_Weights

class Inception(BaseModel):
    def __init__(self, num_classes=10):
        super(Inception, self).__init__(num_classes)
        # Use the updated API with weights
        weights = Inception_V3_Weights.IMAGENET1K_V1  # or use Inception_V3_Weights.DEFAULT for the most up-to-date weights
        self.model = models.inception_v3(weights=weights)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        # Forward pass through the inception model
        outputs = self.model(x)
        if self.training:
            # If the model is in training mode, return only the main output
            return outputs.logits
        else:
            # Otherwise, return the outputs directly (for evaluation mode)
            return outputs