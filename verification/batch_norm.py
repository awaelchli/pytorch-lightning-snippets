import torch
from torch import nn as nn
from typing import Tuple, List

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
        self._detected_pairs = []
        self._module_sequence = []

    @property
    def detected_pairs(self) -> List[Tuple]:
        return self._detected_pairs

    def check(self, input_array=None):
        input_array = self._get_input_array_copy(input_array)
        hook_handles = self._register_hooks()

        # trigger the hooks and collect sequence of layers
        self._model_forward(input_array)

        for hook in hook_handles:
            hook.remove()

        self._detected_pairs = []
        for (name0, mod0), (name1, mod1) in zip(self._module_sequence[:-1], self._module_sequence[1:]):
            bias = getattr(mod0, "bias", None)
            detected = (
                isinstance(mod1, self.normalization_layer)
                and mod1.training  # TODO: do we want/need this check?
                and isinstance(bias, torch.Tensor)
                and bias.requires_grad
            )
            if detected:
                self._detected_pairs.append((name0, name1))

        return not self._detected_pairs

    def _create_hook(self, module_name):
        def hook(module, inp_, out_):
            self._module_sequence.append((module_name, module))
        return hook

    def _register_hooks(self):
        hook_handles = []
        for name, module in self.model.named_children():
            handle = module.register_forward_hook(self._create_hook(name))
            hook_handles.append(handle)
        return hook_handles


class BatchNormVerificationCallback(VerificationCallbackBase):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def message(self, detections: List[Tuple]):
        first_detection = detections[0]
        message = (
            f"Detected a layer '{first_detection[0]}' with bias followed by"
            f" a normalization layer '{first_detection[1]}'."
            f" This makes the normalization ineffective and can lead to unstable training. "
            f" Either remove the normalization or turn off the bias."
        )
        return message

    def on_train_start(self, trainer, pl_module):
        verification = BatchNormVerification(pl_module)
        result = verification.check(
            input_array=pl_module.example_input_array,
        )
        if not result:
            self._raise(verification.detected_pairs)
