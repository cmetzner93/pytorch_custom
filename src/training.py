"""
This file contains source code for the Trainer class to control the training
procedure of a machine learning model implemented in pytorch.
    @author: Christoph S. Metzner
    @date created:  06/18/2024
    @last modified: 06/19/2024
"""

# Load libraries

# built-in
import os
import time
from typing import Dict, Union

# installed
import numpy as np
import torch
import torch.distributed as dist
import toch.nn.functional as F


class Trainer:

    def __init__(
        self,
        model,
        train_kwargs: Dict[str, Union[int, float, bool]],
        paths_dict: Dict[str, str],
        ddp_training: bool = False,
        device: str = None,
    )
    self._model = model
    self._paths_dict = paths_dict
    self._ddp_training = ddp_training
    self._device = device

    # Initialize optimizer, scaler, scheduler, and loss function
    self._optimizer = torch.optim.AdamW(self._model.parameters(), lr=train_kwargs['learning_rate'], betas=(0.9, 0.999))
    self._scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=self._optimizer, total_iters=5)
    self._scaler = torch.cuda.amp.GradScaler(enabled=True)
    self._loss_fct = torch.nn.CrossEntropyLoss()
    
    # Initialize early stopping parameters
    self._best_val_loss = np.inf
    self._patience = train_kwargs['patience']
    self._patience_counter = 0

    def training(self, train_loader, val_loader):
        for epoch in range(self.curr_epoch, self._epochs):
            start_time_epoch = time.time()
            print(f'Current epoch: {epoch + 1}', flush=True)

            # Set the model to training mode
            self._model.train(True)

            if self._ddp_training:
                train_loader.sampler.set_epoch(epoch)

            start_time = time.time()
            train_loss = self._train_one_epoch(train_loader=train_loader)
            end_time = time.time()

            print(f'Current training loss: {train_loss}', flush=True) 
            print(f'Training time: {end_time - start_time:.2f}', flush=True)

            self._scheduler.step()
            start_time = time.time()
            val_loss = self._validate(val_loader=val_loader)
            if self._ddp_training:
                dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
            val_loss = val_loss.detach().cpu().numpy()

            end_time = time.time()
            print(f'Current best validation loss: {self._best_val_loss}', flush=True)
            print(f'Current validation loss: {val_loss}', flush=True)

            # Check for early stopping condition
            early_stopping = _early_stopping(val_loss=val_loss)
            if early_stopping:
                break

    def _train_one_epoch(self, train_loader) -> torch.Tensor:
        for b, batch in enumerate(train_loader):
            if self._debugging:
                if b + 1 == 2:
                    break

            X = batch['X'].to(self._device, non_blocking=True)
            Y = batch['Y'].to(self._device, non_blocking=True)
            # A = batch['A'].to(self._device, non_blocking=True)  # Transformer Model

           with torch.cuda.amp.autocast(enabled=True):
                if self._model_type in ['CLF']:
                    logits = self._model(X, A)
                else:
                    logits = self._model(X)

                train_loss = self._loss_fct(logits, Y)
            self._scaler.scale(train_loss).backward()

            self._scaler.step(self._optimizer)
            self._scaler.update()
            self._optimizer.zero_grad(set_to_none=True)
        return train_loss.detach().cpu().numpy()




    def _early_stopping(self, val_loss: torch.Tensor) -> bool:
        if val_loss < self._best_val_loss:
            self._best_val_loss = val_loss
            self._patience_counter = 0
            torch.save(self._model.state_dict(), self._paths_dict['model_path'])
        else:
            self._patience_counter += 1
            if self._patience_counter >= self._patience:
                print(f'Patience == Patience Counter', flush=True)
                print('Apply early stopping of model training', flush=True)
                return True
        return False














