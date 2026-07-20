from pathlib import Path
from typing import Tuple, List
import cv2
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader, Subset

class ImageDataset(Dataset):
    """
    PyTorch dataset for loading labeled frame files.

    Args:
        imgs: Sequence of image file paths.
        labels: Sequence of labels corresponding to ``imgs``.
        dtype: The torch dtype of the image tensors.
        channels_first: If ``True``, returns images in ``(C, H, W)`` format;
            otherwise, ``(H, W, C)``. 
        transform: Optional transform applied to each image.
    """
    def __init__(self, imgs: List[Path], labels: torch.Tensor, dtype: torch.dtype,
                 channels_first: bool = True, transform = None):
        super().__init__()
        self.imgs = imgs
        self.labels = labels
        self.transform = transform
        self.channels_first = channels_first
        self.dtype = dtype

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        img = cv2.cvtColor(cv2.imread(self.imgs[idx]), cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0

        if self.channels_first:
            img = np.transpose(img, (2, 0, 1))
        img = torch.tensor(img, dtype = self.dtype)

        if self.transform:
            img = self.transform(img)
    
        return  img, self.labels[idx]