import torch
from torch import nn as nn
from typing import Tuple, List, Callable

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
        self._hook_handles = []
        self._module_sequence = []
        self._detected_pairs = []

    @property
    def detected_pairs(self) -> List[Tuple]:
        return self._detected_pairs

    def check(self, input_array=None) -> bool:
        input_array = self._get_input_array_copy(input_array)
        self.register_hooks()
        # trigger the hooks and collect sequence of layers
        self._model_forward(input_array)
        self.destroy_hooks()
        self.collect_detections()
        return not self._detected_pairs

    def collect_detections(self):
        detected_pairs = []
        for (name0, mod0), (name1, mod1) in zip(
            self._module_sequence[:-1], self._module_sequence[1:]
        ):
            bias = getattr(mod0, "bias", None)
            detected = (
                isinstance(mod1, self.normalization_layer)
                and mod1.training  # TODO: do we want/need this check?
                and isinstance(bias, torch.Tensor)
                and bias.requires_grad
            )
            if detected:
                detected_pairs.append((name0, name1))
        self._detected_pairs = detected_pairs
        return detected_pairs

    def register_hooks(self):
        hook_handles = []
        for name, module in self.model.named_modules():
            handle = module.register_forward_hook(self._create_hook(name))
            hook_handles.append(handle)
        self._hook_handles = hook_handles

    def _create_hook(self, module_name) -> Callable:
        def hook(module, inp_, out_):
            self._module_sequence.append((module_name, module))

        return hook

    def destroy_hooks(self):
        for hook in self._hook_handles:
            hook.remove()
        self._hook_handles = []


class BatchNormVerificationCallback(VerificationCallbackBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._verification = None

    def message(self, detections: List[Tuple]) -> str:
        first_detection = detections[0]
        message = (
            f"Detected a layer '{first_detection[0]}' with bias followed by"
            f" a normalization layer '{first_detection[1]}'."
            f" This makes the normalization ineffective and can lead to unstable training."
            f" Either remove the normalization or turn off the bias."
        )
        return message

    def on_train_start(self, trainer, pl_module):
        self._verification = BatchNormVerification(pl_module)
        self._verification.register_hooks()

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        if batch_idx > 0:
            return
        detected_pairs = self._verification.collect_detections()
        if detected_pairs:
            self._raise(detections=detected_pairs)
        self._verification.destroy_hooks()
