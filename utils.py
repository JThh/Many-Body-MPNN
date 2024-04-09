import torch
import torch.optim as optim
from torch.nn.functional import mse_loss

from tqdm.auto import tqdm
import logging


def compute_loss(pred, target):
    return mse_loss(pred.squeeze(), target)

def evaluate(model, test_loader, device):
    model.eval()
    model.to(device)
    total_loss = 0
    with torch.no_grad():
        for data in tqdm(test_loader, desc="Evaluating", leave=False):
            data = data.to(device)
            out = model(data)
            loss = compute_loss(out, data.y)
            total_loss += loss.item() * data.num_graphs
    avg_loss = total_loss / len(test_loader.dataset)
    print(f'Average Test Loss: {avg_loss:.4f}')
    logging.info(f'Average Test Loss: {avg_loss:.4f}')
    return avg_loss

def train(model, train_loader, val_loader, device, n_epochs=50, lr=0.005, return_full_losses=False):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    model.to(device)  # Move model to the appropriate device
    if return_full_losses:
        train_losses = []
        val_losses = []

    for epoch in tqdm(range(1, n_epochs + 1), desc="Training Epochs"):
        epoch_loss = 0
        for data in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            # print(data.x)
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = compute_loss(out, data.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            epoch_loss += loss.item() * data.num_graphs

        # Log training loss and validate the model's performance
        train_loss = epoch_loss / len(train_loader.dataset)
        val_loss = evaluate(model, val_loader, device)
        tqdm.write(f'Epoch: {epoch}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
        logging.info(f'Epoch: {epoch}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
        if return_full_losses:
            train_losses.append(train_loss)
            val_losses.append(val_loss)

    if return_full_losses:
        return train_losses, val_losses
    else:
        return train_loss, val_loss
