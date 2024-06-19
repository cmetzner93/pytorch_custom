"""
This file contains source code for the utility function used in this codebase.
    @author: Christoph S. Metzner
    @date: 06/17/2024
    @last modified: 06/18/2024
"""

# Load libraries
# built-in
import os
import sys
from typing import Dict

# installed
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


def compute_performance_metrics(
    y_trues: np.array,
    y_preds: np.array,
    y_probs: np.array = None,
    num_classes: int
    ) -> Dict[str, float]:

    # F1-scores
    f1_macro = f1_score(y_true=y_trues, y_pred=y_preds, average='macro', labels=np.arange(0, num_classes))
    f1_micro = f1_score(y_true=y_trues, y_pred=y_preds, average='micro', labels=np.arange(0, num_classes))

    accuracy = accuracy_score(y_true=y_trues, y_pred=y_preds, normalize=True)

    metrics = {
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
        'f1_ind': f1_ind,
        'accuracy': accuracy
    }

    return metrics

