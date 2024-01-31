import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, log_loss
from sklearn.preprocessing import label_binarize

class BaseModel(nn.Module):
    def __init__(self, num_classes=10):
        super(BaseModel, self).__init__()
        self.model = None
        self.num_classes = num_classes

    def forward(self, x):
        x = self.model(x)
        return x

    def train_model(self, trainloader, criterion, optimizer):
        self.train()  # Set the model to training mode
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()  # Zero the parameter gradients

            outputs = self(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Optimize

            running_loss += loss.item()
        return running_loss
    
    def evaluate(self, testloader):
        self.eval()  # Set the model to evaluation mode
        all_labels = []
        all_predictions = []
        all_probabilities = []

        with torch.no_grad():
            for data in testloader:
                images, labels = data
                outputs = self(images)
                _, predicted = torch.max(outputs.data, 1)
                probabilities = torch.nn.functional.softmax(outputs.data, dim=1)
                all_labels.extend(labels)
                all_predictions.extend(predicted)
                all_probabilities.extend(probabilities)

        # Convert lists to numpy arrays
        all_labels = np.array(all_labels)
        all_predictions = np.array(all_predictions)
        all_probabilities = np.stack([p.numpy() for p in all_probabilities])

        # Compute metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, average='macro')
        recall = recall_score(all_labels, all_predictions, average='macro')
        f1 = f1_score(all_labels, all_predictions, average='macro')
        confusion = confusion_matrix(all_labels, all_predictions)
        auc_roc = roc_auc_score(label_binarize(all_labels, classes=np.unique(all_labels)), all_probabilities, multi_class='ovr')
        logloss = log_loss(all_labels, all_probabilities)

        return accuracy, precision, recall, f1, confusion, auc_roc, logloss

    def set_model(self, model):
        self.model = model
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, self.num_classes)