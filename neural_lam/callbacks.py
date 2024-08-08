# -*- coding: utf-8 -*-
"""Neural-LAM: Graph-based neural weather prediction models for Limited Area Modeling

Module defining customized callbacks.
"""
import sys
import pytorch_lightning as pl
from typing import Any


class LogfilefriendlyProgressBar(pl.callbacks.ProgressBar):
    """Custom progress bar more friendly with non-interactive environments.
    
    
    Parameters
    ----------
    refresh_rate: int or float, default=0.05
        Tells how frequently the bar is updated. If an integer is provided,
        it is taken as the number of batch (e.g. refresh_rate=100 will update
        the progress bar every 100 batch). If a float in provided (between
        0 and 1), it is taken as the proportion of the total number of batches
        (e.g. refresh_rate=0.1 will update the progress bar at each 10% of
        the total number of batches).
    
    bar_length: int, default=50
        The total number of characters in the progress bar
    """
    def __init__(self, refresh_rate = 0.05, bar_length=50):
        super().__init__()
        self.refresh_rate = refresh_rate
        self.bar_length = bar_length
        self.enable = True

    def disable(self):
        self.enable = False

    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str) -> None:
        super().setup(trainer, pl_module, stage)
    
    def on_sanity_check_start(self, *_: Any) -> None:
        super().on_sanity_check_start(*_)
        sys.stdout.flush()
        sys.stdout.write(f"\nSanity checks (one validation pass)...\n")

    def on_sanity_check_end(self, *_: Any) -> None:
        super().on_sanity_check_start(*_)
        sys.stdout.flush()
        sys.stdout.write(f"Sanity checks done.\n")

    def on_train_start(self, *_):
        super().on_train_start(*_)
        # Update refresh_rate
        if isinstance(self.refresh_rate, float):
            self.refresh_rate = int(self.total_train_batches * self.refresh_rate)
        elif isinstance(self.refresh_rate, int):
            pass
        else:
            raise TypeError(f"Incorrect type for argument 'refresh_rate': {type(self.refresh_rate)}. Expect int of float")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        if (batch_idx + 1) % self.refresh_rate == 0:
            percent = (batch_idx / self.total_train_batches) * 100
            n_done = int(percent * self.bar_length / 100)
            bar = "="*n_done + " "*(self.bar_length - n_done)
            metrics = ", ".join([f"{k}={v}" for k,v in self.get_metrics(trainer, pl_module).items()])
            sys.stdout.flush()
            sys.stdout.write(f"Train {trainer.current_epoch}/{trainer.max_epochs}:\t [{bar}] ({percent:.01f}% complete) {metrics}\n")

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)
        metrics = ", ".join([f"{k}={v}" for k,v in self.get_metrics(trainer, pl_module).items()])
        sys.stdout.flush()
        sys.stdout.write(f"Epoch {trainer.current_epoch}/{trainer.max_epochs} done: {metrics}\n\n")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx)  # don't forget this :)
        if (batch_idx + 1) % self.refresh_rate == 0:
            percent = (batch_idx / self.total_val_batches) * 100
            n_done = int(percent * self.bar_length / 100)
            bar = "-"*n_done + " "*(self.bar_length - n_done)
            metrics = ", ".join([f"{k}={v}" for k,v in self.get_metrics(trainer, pl_module).items()])
            sys.stdout.flush()
            sys.stdout.write(f"Validation:\t [{bar}] ({percent:.01f}% complete) {metrics}\n")

