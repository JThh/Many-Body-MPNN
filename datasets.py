from torch_geometric.datasets import ZINC
from torch_geometric.loader import DataLoader


def load_zinc_dataset_subset(batch_size=32, subset_size=1000):
    # Load the full datasets
    train_dataset_full = ZINC(root='data/ZINC', split='train')
    val_dataset_full = ZINC(root='data/ZINC', split='val')
    test_dataset_full = ZINC(root='data/ZINC', split='test')

    # Select a subset of each dataset
    train_dataset = train_dataset_full[:subset_size]
    val_dataset = val_dataset_full[:subset_size]
    test_dataset = test_dataset_full[:subset_size]

    # Create DataLoaders for each subset
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader