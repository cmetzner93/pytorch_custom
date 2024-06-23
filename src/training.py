"""
This file contains source code for the Trainer class to control the training
procedure of a machine learning model implemented in pytorch.
    @author: Christoph S. Metzner
    @date created:  06/18/2024
    @last modified: 06/23/2024
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
import torch.nn.functional as F


class Trainer:

    def __init__(
        self,
        model,
        model_type: str,
        model_name: str, 
        train_kwargs: Dict[str, Union[int, float, bool]],
        paths_dict: Dict[str, str],
        debugging: bool = False,
        ddp_training: bool = False,
        checkpoint: Dict[str, torch.Tensor] = None,
        device: str = None,
    ):
        self._model = model
        self._model_type = model_type
        self._model_name = model_name
        self._paths_dict = paths_dict
        self._debugging = debugging
        self._ddp_training = ddp_training
        self._device = device

        # Initialize optimizer, scaler, scheduler, and loss function
        self._optimizer = torch.optim.AdamW(self._model.parameters(), lr=train_kwargs['learning_rate'], betas=(0.9, 0.999))
        self._scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=self._optimizer, total_iters=5)
        self._mixed_precision = False
        if self._device in ['cpu', 'cuda']:
            self._mixed_precision = True
        self._scaler = torch.cuda.amp.GradScaler(enabled=self._mixed_precision)
        self._loss_fct = torch.nn.CrossEntropyLoss()

        # Mixed Precision
        mixed_precision = False
        if self._device in ['cpu', 'cuda']:
            mixed_precision = True
        self._epochs = train_kwargs['epochs']
        self._curr_epoch = 0
    
        # Initialize early stopping parameters
        self._best_val_loss = np.inf
        self._patience = train_kwargs['patience']
        self._patience_counter = 0

        if checkpoint is not None:
            self._model.load_state_dict(checkpoint['model_state_dict'])
            self._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self._scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self._scaler.load_state_dict(checkpoint['scaler_state_dict'])

            self._curr_epoch = checkpoint['epoch']
            self._best_val_loss = checkpoint['best_val_loss']
            self._patience_counter = checkpoint['patience_counter']

    def training(self, train_loader, val_loader):
        for epoch in range(self._curr_epoch, self._epochs):
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
            early_stopping = self._early_stopping(val_loss=val_loss)
            if early_stopping:
                break

    def _train_one_epoch(self, train_loader) -> torch.Tensor:
        for b, batch in enumerate(train_loader):
            print(f'Current batch: {b}', flush=True)
            if self._debugging:
                if b + 1 == 2:
                    break

            X = batch['X'].to(self._device, non_blocking=True)
            Y = batch['Y'].to(self._device, non_blocking=True)
            # A = batch['A'].to(self._device, non_blocking=True)  # Transformer Model

            with torch.cuda.amp.autocast(enabled=self._mixed_precision):
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

    def _validate(self, val_loader) -> torch.Tensor:
        losses = torch.zeros(len(val_loader), device=self._device)
        self._model.eval()
        with torch.no_grad():
            for b, batch in enumerate(val_loader):
                if self._debugging:
                    if b + 1 == 2:
                        break
                X = batch['X'].to(self._device, non_blocking=True)
                Y = batch['Y'].to(self._device, non_blocking=True)
                # A = batch['A'].to(self._device, non_blocking=True)  # Transformer Model

                with torch.cuda.amp.autocast(enabled=self._mixed_precision):
                    if self._model_type in ['CLF']:
                        logits = self._model(X, A)
                    else:
                        logits = self._model(X)
                    val_loss = torch.zeros(1, device=self._device)
                    val_loss = self._loss_fct(logits, Y)
                    losses[b] = val_loss.detach()

        return torch.mean(losses)

    def _predict(self, inf_loader) -> Dict['str', Union[float, np.array]]:
        y_trues = []
        y_probs = []
        y_probs_all = []
        y_preds = []
        indices_array = []
        running_tloss = torch.zeros(1, device=self._device)
        self._model.eval()
        with torch.no_grad():
            for b, batch in enumerate(inf_loader):
                if self._debugging:
                    if b + 1 == 2:
                        break
                X = batch['X'].to(self._device, non_blocking=True)
                Y = batch['Y'].to(self._device, non_blocking=True)
                # A = batch['A'].to(self._device, non_blocking=True)  # Transformer Model

                with torch.cuda.amp.autocast(enabled=self._mixed_precision):
                    if self._model_type in ['CLF']:
                        logits = self._model(X, A)
                    else:
                        logits = self._model(X)

                tloss = self._loss_fct(logits, Y)
                running_tloss += tloss

                soft_out = F.softmax(logits, dim=-1)
                soft_out_max = soft_out.max(-1)
                probs = soft_out_max[0].detach().cpu().numpy()
                probs_idx = soft_out_max[1].detach().cpu().numpy()
                y_probs.extend(probs)
                y_probs_all.extend(soft_out.detach().cpu().numpy())
                y_preds.extend(probs_idx)
                y_trues.extend(Y.detach().cpu().numpy())
                indices_array.extend(batch['idx'].detach().cpu().numpy())

        scores = {
            'y_trues': np.array(y_trues),
            'y_preds': np.array(y_preds),
            'y_probs': np.array(y_probs),
            'y_probs_all': np.array(y_probs_all),
            'indices': np.array(indices_array),
            'test_loss': (running_tloss / (b + 1)).detach().cpu().numpy()
        }
        return scores


    def _early_stopping(self, val_loss: torch.Tensor) -> bool:
        if val_loss < self._best_val_loss:
            self._best_val_loss = val_loss
            self._patience_counter = 0
            torch.save(self._model.state_dict(), os.path.join(self._paths_dict['path_models'], f'{self._model_name}.pt'))
        else:
            self._patience_counter += 1
            if self._patience_counter >= self._patience:
                print(f'Patience == Patience Counter', flush=True)
                print('Apply early stopping of model training', flush=True)
                return True
        return False

    def create_checkpoint(self, epoch: int) -> None:
        torch.save(
            {
                'epoch': epoch + 1,
                'model_state_dict': self._model.state_dict(),
                'optimizer_state_dict': self._optimizer.state_dict(),
                'scaler_state_dict': self._scaler.state_dict(),
                'scheduler_state_dict': self._scheduler.state_dict(),
                'best_val_loss': self._best_val_loss,
                'patience_counter': self._patience_counter
            },
            os.path.join(self._paths_dict['path_models'], f'{self._model_name}_checkpoint.tar')
        )
