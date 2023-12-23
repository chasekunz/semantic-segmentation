#!/usr/bin/env python3
import torch

def train_model(model, trainloader, validateloader, criterion, optimizer, num_epochs=10, patience=3, device='cuda'):
    """
    Trains a given model using the provided data loaders, criterion, optimizer, and device.

    Args:
        model (torch.nn.Module): The model to be trained.
        trainloader (torch.utils.data.DataLoader): The data loader for training data.
        validateloader (torch.utils.data.DataLoader): The data loader for validation data.
        criterion (torch.nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer.
        num_epochs (int, optional): The number of epochs to train the model (default: 10).
        patience (int, optional): The number of epochs to wait for improvement in validation loss before early stopping (default: 3).
        device (str, optional): The device to use for training (default: 'cuda').

    Returns:
        list: A list of loss values for each epoch.
    """
    # Initialize lists to store the loss values for each epoch
    train_losses = []
    valid_losses = []
    
    # Initialize variables for early stopping
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        for images, masks in trainloader:
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)['out']
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        average_train_loss = total_train_loss / len(trainloader)
        train_losses.append(average_train_loss)

        # Validation phase
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for images, masks in validateloader:
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)['out']
                loss = criterion(outputs, masks)
                total_val_loss += loss.item()
        average_val_loss = total_val_loss / len(validateloader)
        valid_losses.append(average_val_loss)

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {average_train_loss:.4f}, Validation Loss: {average_val_loss:.4f}')

        # Check for improvement for early stopping
        if average_val_loss < best_val_loss:
            best_val_loss = average_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping initiated.")
                break

    return train_losses, valid_losses
