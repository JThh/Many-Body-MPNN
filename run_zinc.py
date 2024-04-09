import logging
import os
from datetime import datetime

import torch

from datasets import load_zinc_dataset_subset
from models import ManyBodyMPNN, GCN, ChebNet
from utils import compute_loss, evaluate, train


def main():
    in_channels = 1
    hidden_channels = 8
    output_channels = 1
    num_of_layers = 2
    max_order = 5
    edge_feature_dim = 1
    K = 4  # Chebyshev filter size
    n_epochs = 10

    batch_size = 32
    subset_size = 200000 
    train_loader, val_loader, test_loader = load_zinc_dataset_subset(batch_size, subset_size)
    
    
    models = {
        # "GCN": GCN(in_channels, hidden_channels, output_channels, num_of_layers),
        # "ChebNet": ChebNet(in_channels, hidden_channels, output_channels, num_of_layers, K),
        "ManyBodyMPNN": ManyBodyMPNN(in_channels, hidden_channels, output_channels, num_of_layers,
                                     max_order, edge_feature_dim, K),
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Check if CUDA is available

    # Logging configuration
    formatted_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    dir_path = "./logs"

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(f"{dir_path}/{formatted_time}_training_nepo_{n_epochs}_nlayers_{num_of_layers}_models_{'_'.join(models.keys())}.log", mode='w'),
                                  logging.StreamHandler()])


    for name, model in models.items():
        model.to(device) 
        logging.info(f"Training {name}")
        train_loss, val_loss = train(model, train_loader, val_loader, device, n_epochs=n_epochs, lr=0.005)
        test_loss = evaluate(model, test_loader, device)
        logging.info(f"{name} - Final Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Test Loss: {test_loss:.4f}")
    

if __name__ == "__main__":
    main()





