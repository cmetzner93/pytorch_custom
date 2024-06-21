"""
This file contains the model suite training class object allowing full control over loading, training, and infering a specific model.
    @author:         Christoph S. Metzner
    @date:           06/17/2024
    @last modified:  06/20/2024
"""


# Load libaries
# built-in
import os
import sys

# installed
import torch

# custom
from models import CNN

# Set path to root project
try:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    root = os.path.dirname(os.getcwd())
sys.path.append(root)



def main():
	return 0

def init_model(
    self,
    model_kwargs: Dict
):
    if self._model_type == 'CNN':
        model = CNN(**mmodel_kwargs, device=self._device)
    else:
        raise Exception('Invalid model type!')

    return model




if __name__ == "__main__":
    main()

