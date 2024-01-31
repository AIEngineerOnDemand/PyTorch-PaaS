import torch
import torch.nn as nn
import torch.nn.functional as F
    
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, log_loss
from sklearn.preprocessing import label_binarize

import numpy as np

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)  # 3 input channels, 6 output channels, 5x5 kernel
        self.pool = nn.MaxPool2d(2, 2)  # 2x2 max pooling
        self.conv2 = nn.Conv2d(6, 16, 5)  # 6 input channels, 16 output channels, 5x5 kernel
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # Fully connected layer (16*5*5 inputs, 120 outputs)
        self.fc2 = nn.Linear(120, 84)  # Fully connected layer (120 inputs, 84 outputs)
        self.fc3 = nn.Linear(84, 10)  # Fully connected layer (84 inputs, 10 outputs)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # First conv layer -> ReLU -> Pooling
        x = self.pool(F.relu(self.conv2(x)))  # Second conv layer -> ReLU -> Pooling
        x = x.view(-1, 16 * 5 * 5)  # Flatten the tensor
        x = F.relu(self.fc1(x))  # First fully connected layer -> ReLU
        x = F.relu(self.fc2(x))  # Second fully connected layer -> ReLU
        x = self.fc3(x)  # Third fully connected layer
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
        all_probabilities = np.array(all_probabilities)

        # Compute metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, average='macro')
        recall = recall_score(all_labels, all_predictions, average='macro')
        f1 = f1_score(all_labels, all_predictions, average='macro')
        confusion = confusion_matrix(all_labels, all_predictions)
        auc_roc = roc_auc_score(label_binarize(all_labels, classes=np.unique(all_labels)), all_probabilities, multi_class='ovr')
        logloss = log_loss(all_labels, all_probabilities)

        return accuracy, precision, recall, f1, confusion, auc_roc, logloss