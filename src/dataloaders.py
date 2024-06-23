"""
This file contains source code for multiple dataloaders utilized
    @author: Christoph S. Metzner
    @date: 06/18/2024
    @last modified: 06/22/2024
"""
# built-in
from typing import Dict
# Installed libraries
import torch
from torch.utils.data import Dataset


class GenericDataloader(Dataset):
    """
    A class to load a pytorch dataloader object.

    Attributes
    ----------
    X : torch.tensor
        Integer-indexed text sequences
    Y : torch.tensor
        Ground-truth label corresponding to each sequence


    """

    def __init__(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        self._X = X
        self._Y = Y

    def __len__(self) -> int:
        return len(self._X)

    def __getitem__(self, idx: int) -> Dict[str, torch.tensor]:
        sample = {
            'X': torch.tensor(self._X[idx], dtype=torch.long),
            'Y': torch.tensor(self._Y[idx], dtype=torch.long),
            'idx': torch.tensor(idx, dtype=torch.int32)
        }
        return sample 


class GenericTransformerDataloader(Dataset):
    """
    A class to load a pytorch dataloader object.

    Attributes
    ----------
    X : torch.tensor
        Integer-indexed text sequences
    Y : torch.tensor
        Ground-truth label corresponding to each sequence
    A : torch.tensor
        Attention mask corresponding to padding tokens


    """

    def __init__(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        self._X = X
        self._Y = Y
        self._A = A

    def __len__(self) -> int:
        return len(self._X)

    def __getitem__(self, idx: int) -> Dict[str, torch.tensor]:
        sample = {
            'X': torch.tensor(self._X[idx], dtype=torch.long),
            'Y': torch.tensor(self._Y[idx], dtype=torch.long),
            'A': torch.tensor(self._A[idx], dtype=torch.long),
            'idx': torch.tensor(idx, dtype=torch.int32)
        }
        return sample 
