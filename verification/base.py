from abc import abstractmethod
from copy import deepcopy

import torch.nn as nn

from pytorch_lightning.utilities import rank_zero_warn


class ModelVerificationBase:

    def __init__(self, model: nn.Module, warn: bool = True, error: bool = False):
        super().__init__()
        self.model = model
        self.warn = warn
        self.error = error

    def assert_pass(self, *args, **kwargs) -> bool:
        pass

    @abstractmethod
    def check(self, *args, **kwargs) -> bool:
        pass

    @abstractmethod
    def warning_message(self, *args, **kwargs):
        pass

    def error_message(self, *args, **kwargs):
        return self.warning_message(*args, **kwargs)

    def _raise(self, *args, **kwargs):
        if self.error:
            raise RuntimeError(self.error_message(*args, **kwargs))
        if self.warn:
            rank_zero_warn(self.warning_message(*args, **kwargs))

    def _get_input_array_copy(self, input_array=None):
        input_array = input_array if input_array is not None else getattr(self.model, 'example_input_array', None)
        return deepcopy(input_array)

