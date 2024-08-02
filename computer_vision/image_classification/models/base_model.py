import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, log_loss
from sklearn.preprocessing import label_binarize
import logging

class BaseModel(nn.Module):
    def __init__(self, num_classes=10):
        super(BaseModel, self).__init__()
        self.model = None
        self.num_classes = num_classes

    def forward(self, x):
        x = self.model(x)
        return x

    def train_model(self, trainloader, criterion, optimizer, epochs):
        """
        Train the model using the provided dataloader, criterion, and optimizer.

        Args:
            trainloader (DataLoader): DataLoader for the training data.
            criterion (torch.nn.Module): Loss function.
            optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
            epochs (int): Number of epochs to train the model.

        Returns:
            list: A list of average losses for each epoch.

        Steps:
            1. Set the model to training mode.
            2. Loop over the number of epochs.
            3. For each epoch, loop over the training data.
            4. Zero the parameter gradients.
            5. Perform the forward pass to get model outputs.
            6. Compute the loss using the criterion.
            7. Perform the backward pass to compute gradients.
            8. Update the model parameters using the optimizer.
            9. Log the loss for each epoch.
        """
        self.train()  # Set the model to training mode
        epoch_losses = []
        for epoch in range(epochs):
            running_loss = 0.0
            for inputs, labels in trainloader:
                optimizer.zero_grad()  # Zero the parameter gradients

                outputs = self(inputs)  # Forward pass
                loss = criterion(outputs, labels)  # Compute loss
                loss.backward()  # Backward pass
                optimizer.step()  # Optimize parameters

                running_loss += loss.item()
            average_loss = running_loss / len(trainloader)
            epoch_losses.append(average_loss)
            logging.info(f'Epoch {epoch+1}, Loss: {average_loss}')
        
        return epoch_losses
    
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