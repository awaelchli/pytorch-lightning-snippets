from typing import Any, Sequence, Dict

import numpy as np
import torch
import wandb
from torch import Tensor

from pytorch_lightning import Callback
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_warn
from pytorch_lightning.utilities.apply_func import apply_to_collection


class DataMonitorBase(Callback):

    supported_loggers = (
        TensorBoardLogger,
        WandbLogger,
    )

    def __init__(self, row_log_interval: int = None):
        """
        Base class for monitoring data histograms in a LightningModule.
        This requires a logger configured in the Trainer, otherwise no data is logged.
        The specific class that inherits from this base defines what data gets collected.

        Args:
            row_log_interval: The interval at which histograms should be logged. This defaults to the
                interval defined in the Trainer. Use this to override the Trainer default.
        """
        super().__init__()
        self._row_log_interval = row_log_interval
        self._log = False
        self._trainer = None
        self._train_batch_idx = None

    def on_train_start(self, trainer, pl_module):
        self._log = self._is_logger_available(trainer.logger)
        self._row_log_interval = self._row_log_interval or trainer.row_log_interval
        self._trainer = trainer

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        self._train_batch_idx = batch_idx

    def log_histograms(self, batch, group="") -> None:
        """
        Logs the histograms at the interval defined by `row_log_interval`, given a logger is available.

        Args:
            batch: torch or numpy arrays, or a collection of it (tuple, list, dict, ...), can be nested.
                If the data appears in a dictionary, the keys are used as labels for the corresponding histogram.
                Otherwise the histograms get labelled with an integer index.
                Each label also has the tensors's shape as suffix.
            group: Name under which the histograms will be grouped.
        """
        if not self._log or (self._train_batch_idx + 1) % self._row_log_interval != 0:
            return

        batch = apply_to_collection(batch, dtype=np.ndarray, function=torch.from_numpy)
        named_tensors = dict()
        collect_and_name_tensors(batch, output=named_tensors, parent_name=group)

        for name, tensor in named_tensors.items():
            self.log_histogram(tensor, name)

    def log_histogram(self, tensor: Tensor, name: str) -> None:
        """
        Override this method to customize the logging of histograms.
        Detaches the tensor from the graph and moves it to the CPU for logging.

        Args:
            tensor: The tensor for which to log a histogram
            name: The name of the tensor as determined by the callback. Example: ``ìnput/0/[64, 1, 28, 28]``
        """
        logger = self._trainer.logger
        tensor = tensor.detach().cpu()
        if isinstance(logger, TensorBoardLogger):
            logger.experiment.add_histogram(
                tag=name,
                values=tensor,
                global_step=self._trainer.global_step
            )

        if isinstance(logger, WandbLogger):
            logger.experiment.log(
                row={name: wandb.Histogram(tensor)},
                commit=False,
                step=self._trainer.global_step
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


def collect_and_name_tensors(data: Any, output: Dict[str, Tensor], parent_name: str = "input") -> None:
    """
    Recursively fetches all tensors in a (nested) collection of data (depth-first search) and names them.
    Data in dictionaries get named by their corresponding keys and otherwise they get indexed by an
    increasing integer. The shape of the tensor gets appended to the name as well.

    Args:
        data: A collection of data (potentially nested).
        output: A dictionary in which the outputs will be stored.
        parent_name: Used when called recursively on a nested input data.

    Example:
        >>> data = {"x": torch.zeros(2, 3), "y": {"z": torch.zeros(5)}, "w": 1}
        >>> output = {}
        >>> collect_and_name_tensors(data, output)
        >>> output  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
        {'input/x/[2, 3]': ..., 'input/y/z/[5]': ...}
    """
    assert isinstance(output, dict)
    if isinstance(data, Tensor):
        name = f"{parent_name}/{shape2str(data)}"
        output[name] = data

    if isinstance(data, dict):
        for k, v in data.items():
            collect_and_name_tensors(v, output, parent_name=f"{parent_name}/{k}")

    if isinstance(data, Sequence) and not isinstance(data, str):
        for i, item in enumerate(data):
            collect_and_name_tensors(item, output, parent_name=f"{parent_name}/{i:d}")


def shape2str(tensor: Tensor) -> str:
    """
    Returns the shape of a tensor in bracket notation as a string.

    Example:
        >>> shape2str(torch.rand(1, 2, 3))
        '[1, 2, 3]'
        >>> shape2str(torch.rand(4))
        '[4]'
    """
    return "[" + ", ".join(map(str, tensor.shape)) + "]"
