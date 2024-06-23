"""
This file contains source code to create a dummy dataset.
    @author: Christoph S. Metzner
    @date: 06/20/2024
    @last modified: 06/22/2024
"""
import os
import json
from typing import Dict, Union, List
import torch
import pandas as pd
def create_dummy_dataset(
    N: Union[int, List[int]],
    path_save_data: str,
    vocab_size: int = 20000
    ) -> None:
    """
    This function creates a dummy dataset containing samples as integer-index
    sequences annotated with either one of three classes (0, 1, 2). This
    dataset is meant to simulate a multi-class text classification task.

    Parameters
    ----------
    N : Union[int, List[int, int, int]]
        Number of samples in each split (i.e., training, validation, testing)
        If N: int all split have the same number of samples
        If N: List[int, int, int] each split can have a different number of
        samples.
    path_save_data : str
        Absolute path to storage location for dummy dataset.

    """
    if not os.path.exists(os.path.dirname(os.path.join(path_save_data, 'dummy_dataset/'))):
        os.makedirs(os.path.dirname(os.path.join(path_save_data, 'dummy_dataset/')))
    splits = ['train', 'val', 'test']
    for split, n in zip(splits, N):
        X = torch.randint(0, vocab_size + 1, (n, 3000)).tolist()  # 0 [inclusive] to vocab_size [exclusive]
        Y = torch.randint(0, 3, (n, )).tolist()

        df = pd.DataFrame.from_dict({'X': X, 'Y': Y})
        df.to_parquet(os.path.join(path_save_data, 'dummy_dataset', f'dummy_{split}.parquet'))

    d = {'num_labels': 3, 'vocab_size': vocab_size}
    with open(os.path.join(path_save_data, 'dummy_dataset', f'dummy_id2label.json'), 'w') as f:
        json.dump(d, f)



create_dummy_dataset(N=[1000, 1000, 1000], path_save_data='/Users/cmetzner/Desktop/coding_projects/pytorch_custom/data')
