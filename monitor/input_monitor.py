from typing import Any, Sequence

import numpy as np
import torch
import wandb
from torch import Tensor

from pytorch_lightning import Callback
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_warn
from pytorch_lightning.utilities.apply_func import apply_to_collection


class InputMonitor(Callback):

    supported_loggers = (
        TensorBoardLogger,
        WandbLogger,
    )

    def __init__(self, expected_range: tuple = None, log: bool = True):
        super().__init__()
        self.expected_range = expected_range
        self._log = log

    def on_train_start(self, trainer, pl_module):
        if self._log:
            self._check_logger(trainer.logger)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        if self._log and (batch_idx + 1) % trainer.row_log_interval == 0:
            self.log_histograms(trainer, batch)

    def _check_logger(self, logger):
        if logger is None:
            rank_zero_warn("Cannot log histograms because Trainer has no logger.")
            return

        if not isinstance(logger, self.supported_loggers):
            rank_zero_warn(
                f"{self.__class__.__name__} does not support logging with {logger.__class__.__name__}."
                f" Supported loggers are: {', '.join(map(lambda x: str(x.__name__), self.supported_loggers))}"
            )
            return

    @staticmethod
    def log_histograms(trainer, batch):
        logger = trainer.logger
        batch = apply_to_collection(batch, dtype=np.ndarray, function=torch.from_numpy)
        named_tensors = dict()
        collect_and_name_tensors(batch, accumulator=named_tensors, parent_name="input")

        for name, tensor in named_tensors.items():
            if isinstance(logger, TensorBoardLogger):
                logger.experiment.add_histogram(
                    tag=name,
                    values=tensor,
                    global_step=trainer.global_step
                )

            if isinstance(logger, WandbLogger):
                logger.experiment.log(
                    row={name: wandb.Histogram(tensor)},
                    commit=False,  # TODO: needed ?
                    step=trainer.global_step,
                )


def collect_and_name_tensors(data, accumulator, parent_name="input"):
    if isinstance(data, Tensor):
        name = f"{parent_name}/{shape2str(data)}"
        accumulator[name] = data

    if isinstance(data, dict):
        for k, v in data.items():
            collect_and_name_tensors(v, accumulator, parent_name=f"{parent_name}/{k}")

    if isinstance(data, Sequence) and not isinstance(data, str):
        for i, item in enumerate(data):
            collect_and_name_tensors(item, accumulator, parent_name=f"{parent_name}/{i:d}")


def shape2str(tensor):
    return "(" + ", ".join(map(str, tensor.shape)) + ")"
