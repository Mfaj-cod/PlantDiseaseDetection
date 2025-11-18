import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from .logging import logger
from .utils import ConvertToRGB, get_mean_std


def load_data(data_dir):
    transform = transforms.Compose(
        [
            ConvertToRGB(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    logger.info(f"Loaded dataset from {data_dir} with {len(dataset)} samples.")
    return dataset



def transform_data(dataset):
    batch_size = 32
    dataset_loader = DataLoader(dataset, batch_size=batch_size)
    mean, std = get_mean_std(dataset_loader)
    transform_normalized = transforms.Compose(
        [
            ConvertToRGB(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    dataset = datasets.ImageFolder(root=dataset.root, transform=transform_normalized)
    logger.info(f"Transformed dataset with normalization.")
    return dataset, dataset_loader



def split_data(dataset, split_ratio=0.8):
    train_size = int(split_ratio * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    logger.info(f"Splitted dataset into {train_size} training and {val_size} validation samples.")

    return train_dataset, val_dataset, train_loader, val_loader