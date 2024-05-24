import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

from PIL import Image


# Path to the CSV file
CSV_PATH = "data/garbage_classification_preprocessed/data.csv"


class GarbageDataset(Dataset):
    def __init__(self):
        # Load the CSV file
        self.df = pd.read_csv(CSV_PATH)

        # Data normalization
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self) -> int:
        """Get the length of the dataset

        Returns:
            int: Length of the dataset
        """
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Get an item from the dataset

        Args:
            idx (int): Index of the item

        Returns:
            tuple[torch.Tensor, int]: Image and label
        """
        img_path, label = self.df.iloc[idx]
        img = Image.open(img_path)
        img = self.transform(img)
        return img, label


def get_dataloader(batch_size: int = 32) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Get DataLoader for the garbage classification dataset

    Args:
        batch_size (int, optional): Batch size. Defaults to 32.

    Returns:
        tuple[DataLoader, DataLoader, DataLoader]: Train, validation, and test DataLoader
    """

    dataset = GarbageDataset()

    # Randomly split the dataset into training, validation, and test sets by 70%, 15%, and 15%
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, _, _ = get_dataloader(batch_size=32)
    img, label = next(iter(train_loader))

    print(f"Image shape: {img.shape}")
    print(f"Label: {label}")
