"""
This file contains source code for the utility function used in this codebase.
    @author: Christoph S. Metzner
    @date: 06/17/2024
    @last modified: 06/20/2024
"""

# Load libraries
# built-in
import os
import sys
from typing import Dict, Union, List

# installed
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import torch

def compute_performance_metrics(
    y_trues: np.array,
    y_preds: np.array,
    num_classes: int
    ) -> Dict[str, Union[float, np.array]]:
    """
    This function computes multiple performance metrics to evaluate a machine 
    learning model.

    Parameters
    ----------
    y_trues : np.array
        Array containing the ground-truth labels corresponding to the samples
    y_preds : np.array
        Array containing the predictions correspendoning to the samples
    num_classes : int
        Number of classes in the label space

    Return
    ------
    Dict[str, Union[float, np.array]]
        Dictionary containing the results the various performance metrics 
    """
    # F1-scores
    f1_macro = f1_score(
        y_true=y_trues,
        y_pred=y_preds,
        average='macro',
        labels=np.arange(0, num_classes)
    )
    f1_micro = f1_score(
        y_true=y_trues,
        y_pred=y_preds,
        average='micro',
        labels=np.arange(0, num_classes)
    )
    f1_ind = f1_score(
        y_true=y_trues,
        y_pred=y_preds,
        average=None,
        labels=np.arange(0, num_classes)
    )

    accuracy = accuracy_score(y_true=y_trues, y_pred=y_preds, normalize=True)

    metrics = {
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
        'f1_ind': f1_ind,
        'accuracy': accuracy
    }

    return metrics


def create_dummy_dataset(
    N: Union[int, List[int, int, int]],
    path_save_data: str
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
    if is_instance(N, int):
        N = [N] * 3
    elif is_instance(N, list):
        assert len(N) == 3

    splits = ['train', 'val', 'test']
    for split, n in zip(splits, N):
        X = torch.randint(0, 20000, (3000, ))
        Y = torch.randint(0, 3, (n, ))

        df = pd.DataFrame.from_dict({'X': X, 'Y': Y})
        df.to_parquet(os.path.join(path_save_data, f'dummy_{split}.parquet'))


