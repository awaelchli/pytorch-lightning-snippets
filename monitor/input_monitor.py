import warnings
from typing import Any, Sequence, Tuple

import numpy as np
import torch
import wandb
from torch import Tensor

from pytorch_lightning import Callback
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_warn
from pytorch_lightning.utilities.apply_func import apply_to_collection


# TODO: docstrings, typehints
class InputMonitor(Callback):

    supported_loggers = (
        TensorBoardLogger,
        WandbLogger,
    )

    def __init__(self, expected_range: Tuple[float, float] = None, log: bool = True):
        super().__init__()
        self.expected_range = expected_range
        self._log = log

    def on_train_start(self, trainer, pl_module):
        self._log = self._log and self._is_logger_available(trainer.logger)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        batch = apply_to_collection(batch, dtype=np.ndarray, function=torch.from_numpy)
        named_tensors = dict()
        collect_and_name_tensors(batch, accumulator=named_tensors, parent_name="input")

        for name, tensor in named_tensors:
            if not self.check_range(tensor):
                warnings.warn(
                    f"Tensor {name} of dtype {tensor.dtype} in batch {batch_idx} is has values"
                    f" outside expected range ({self.expected_range[0]}, {self.expected_range[1]}):"
                    f" min = {tensor.min().item():.4f}, max = {tensor.max().item():.4f}"
                )

        if self._log and (batch_idx + 1) % trainer.row_log_interval == 0:
            self._log_histograms(trainer, batch)

    def check_range(self, tensor: Tensor):
        return tensor.ge(self.expected_range[0]).all() and tensor.le(self.expected_range[1]).all()

    def log_histogram(self, logger: Any, tensor: Tensor, name: str, global_step: int) -> None:
        """ Override to customize logging of histogram. """
        if isinstance(logger, TensorBoardLogger):
            logger.experiment.add_histogram(
                tag=name,
                values=tensor,
                global_step=global_step
            )

        if isinstance(logger, WandbLogger):
            logger.experiment.log(
                row={name: wandb.Histogram(tensor)},
                commit=False,  # TODO: needed ?
                step=global_step
            )

    def _is_logger_available(self, logger) -> bool:
        available = True
        if logger is None:
            rank_zero_warn("Cannot log histograms because Trainer has no logger.")
            available = False
        if not isinstance(logger, self.supported_loggers):
            rank_zero_warn(
                f"{self.__class__.__name__} does not support logging with {logger.__class__.__name__}."
                f" Supported loggers are: {', '.join(map(lambda x: str(x.__name__), self.supported_loggers))}"
            )
            available = False
        return available

    def _log_histograms(self, trainer, batch) -> None:
        logger = trainer.logger


        for name, tensor in named_tensors.items():
            self.log_histogram(logger, tensor, name, trainer.global_step)


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
