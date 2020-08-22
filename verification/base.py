from abc import abstractmethod
from copy import deepcopy

import torch.nn as nn

from pytorch_lightning import Callback
from pytorch_lightning.utilities import rank_zero_warn


class VerificationBase:

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    @abstractmethod
    def check(self, *args, **kwargs) -> bool:
        pass

    def _get_input_array_copy(self, input_array=None):
        input_array = input_array if input_array is not None else getattr(self.model, 'example_input_array', None)
        return deepcopy(input_array)

    def _model_forward(self, input_array):
        if isinstance(input_array, tuple):
            return self.model(*input_array)
        if isinstance(input_array, dict):
            return self.model(**input_array)
        return self.model(input_array)


class VerificationCallbackBase(Callback):

    def __init__(self, warn: bool = True, error: bool = False):
        self._raise_warning = warn
        self._raise_error = error

    @abstractmethod
    def message(self, *args, **kwargs):
        pass

    def warning_message(self, *args, **kwargs):
        return self.message(*args, **kwargs)

    def error_message(self, *args, **kwargs):
        return self.message(*args, **kwargs)

    def _raise(self, *args, **kwargs):
        if self._raise_error:
            raise RuntimeError(self.error_message(*args, **kwargs))
        if self._raise_warning:
            rank_zero_warn(self.warning_message(*args, **kwargs))