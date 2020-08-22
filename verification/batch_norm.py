import torch
from torch import nn as nn

from verification.base import VerificationBase, VerificationCallbackBase


class BatchNormVerification(VerificationBase):

    normalization_layer = (
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.SyncBatchNorm,
        nn.InstanceNorm1d,
        nn.InstanceNorm2d,
        nn.InstanceNorm3d,
        nn.GroupNorm,
        nn.LayerNorm,
    )

    def __init__(self, model: nn.Module):
        super().__init__(model)
        self._module_list = []
        self._detected_pairs = []

    def get_hook(self):
        def hook(m: nn.Module, inp_, out_):
            self._module_list.append(m)
        return hook

    def check(self, input_array=None):
        hook_handles = []
        self._module_list = []
        self._detected_pairs = []
        hook = self.get_hook()
        for module in self.model.modules():
            handle = module.register_forward_hook(hook)
            hook_handles.append(handle)
            input_array = self._get_input_array_copy(input_array)
            self.model(input_array)

        for prev, current in zip(self._module_list[:-1], self._module_list[1:]):
            bias = getattr(prev, "bias", None)
            detected = (isinstance(current, self.normalization_layer)
                        # and current.training  # TODO: do we want/need this check?
                        and isinstance(bias, torch.Tensor)
                        and bias.requires_grad)
            if detected:
                self._detected_pairs.append((prev, current))

        return not self._detected_pairs


class BatchNormVerificationCallback(VerificationCallbackBase):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def message(self):
        message = (
            "Detected a layer with bias followed by a normalization layer."
            " This makes the normalization ineffective and can lead to unstable training. "
            " Turn the bias off on this layer."
        )
        return message

    def on_train_start(self, trainer, pl_module):
        verification = BatchNormVerification(pl_module)
        result = verification.check(
            input_array=pl_module.example_input_array,
        )
        if not result:
            self._raise()
