import logging
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

import torch

from datasets import load_zinc_dataset_subset
from models import ManyBodyMPNN, GCN, ChebNet
from utils import compute_loss, evaluate, train


def main():
    in_channels = 1
    hidden_channels = 16
    output_channels = 1
    num_of_layers = 3
    max_order = 4
    edge_feature_dim = 1
    K = 4  # Chebyshev filter size
    n_epochs = 30
    n_experiments = 10  # Number of experiments to run for each model

    batch_size = 32
    subset_size = 200000
    train_loader, val_loader, test_loader = load_zinc_dataset_subset(batch_size, subset_size)

    models = {
        "ChebNet": lambda : ChebNet(in_channels, hidden_channels, output_channels, num_of_layers, K),
        "ManyBodyMPNN": lambda : ManyBodyMPNN(in_channels, hidden_channels, output_channels, num_of_layers, max_order, edge_feature_dim, K),
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Logging configuration
    formatted_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    dir_path = "./logs"

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(f"{dir_path}/{formatted_time}_training_nepo_{n_epochs}_nlayers_{num_of_layers}_maxorder_{max_order}_cfilt_{K}_models_{'_'.join(models.keys())}.log", mode='w'),
                                  logging.StreamHandler()])


    test_losses = {name: np.zeros((n_experiments, n_epochs)) for name in models}

    for name, model_fn in models.items():
        for experiment in range(n_experiments):
            logging.info(f"Training {name}, Experiment {experiment+1}/{n_experiments}")
            model_instance = model_fn().to(device)
            _, experiment_losses = train(model_instance, train_loader, val_loader, device, n_epochs, lr=0.005, return_full_losses=True)
            test_losses[name][experiment, :] = experiment_losses

    # Plotting
    plt.figure(figsize=(10, 6))
    epochs_range = range(1, n_epochs + 1)
    for name, losses in test_losses.items():
        mean_losses = np.mean(losses, axis=0)
        std_losses = np.std(losses, axis=0)
        plt.plot(epochs_range, mean_losses, label=f'{name} Mean Test Loss')
        plt.fill_between(epochs_range, mean_losses - std_losses, mean_losses + std_losses, alpha=0.1)
    
    plt.xlabel('Epochs')
    plt.ylabel('Test Loss')
    plt.legend()
    plt.title('Test Losses with Variances Across Models and Experiments')
    plt.savefig(f"{formatted_time}_{name}_{experiment}_test_losses_zinc.png")
    plt.show()

if __name__ == "__main__":
    main()



