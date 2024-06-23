"""
This file contains source code for the utility function used in this codebase.
    @author: Christoph S. Metzner
    @date: 06/17/2024
    @last modified: 06/23/2024
"""
# Load libraries
# built-in
import os
import sys
import argparse
from typing import Dict, Union, List

# installed
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

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


def eval_bool_command(arg):
    if isinstance(arg, bool):
        return arg
    if arg.lower() in ('yes', 'true', '1', 'y', 't'):
        return True
    elif arg.lower() in ('no', 'false', '0', 'n', 'f'):
        return False
    else:
        argparse.ArgumentTypeError('Boolean Value Expected')

