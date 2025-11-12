from matplotlib import transforms
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
    transform_normalized = transforms.Compose(
        [
            ConvertToRGB(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=get_mean_std(dataset_loader)[0], std=get_mean_std(dataset_loader)[1]),
        ]
    )
    dataset = datasets.ImageFolder(root=dataset.root, transform=transform_normalized)
    logger.info(f"Transformed dataset with normalization.")
    return dataset, dataset_loader


