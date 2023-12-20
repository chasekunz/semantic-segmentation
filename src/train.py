#!/usr/bin/env python3

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
    # Initialize a list to store the loss values
    epoch_losses = []

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for images, masks in trainloader:
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)['out']
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Average loss for this epoch
        average_loss = total_loss / len(trainloader)
        epoch_losses.append(average_loss)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.4f}')

    return epoch_losses
