from collections import Counter
import pandas as pd
import torch
from tqdm.notebook import tqdm
from .logging import logger


class ConvertToRGB(object):
    def __call__(self, img):
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img
    


def get_mean_std(loader):
    """Computes the mean and standard deviation of image data."""
    channels_sum = torch.zeros(3)
    channels_squared_sum = torch.zeros(3)
    num_batches = 0

    for data, _ in tqdm(loader, desc="Computing mean and std", leave=False):
        channels_sum += data.mean(dim=[0, 2, 3])
        channels_squared_sum += (data ** 2).mean(dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    logger.info(f"Computed mean: {mean}, std: {std}")
    return mean, std



def class_counts(dataset):
    c = Counter(x[1] for x in tqdm(dataset))
    try:
        class_to_index = dataset.class_to_idx
    except AttributeError:
        class_to_index = dataset.dataset.class_to_idx

    logger.info(f"Class counts: {c}")
    return pd.Series({cat: c[idx] for cat, idx in class_to_index.items()})
