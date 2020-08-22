import torch
from torch import nn as nn

from verification.base import ModelVerificationBase


class BatchNormVerification(ModelVerificationBase):

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
        self.module_list = []

    def warning_message(self, *args, **kwargs):
        message = (
            "Detected a layer with bias followed by a normalization layer."
            " This makes the normalization ineffective and can lead to unstable training. "
            " Turn the bias off on this layer."
        )
        return message

    def get_hook(self):
        def hook(m: nn.Module, inp_, out_):
            self.module_list.append(m)
        return hook

    def check(self, input_array=None):
        hook_handles = []
        self.module_list = []
        hook = self.get_hook()
        for module in self.model.modules():
            handle = module.register_forward_hook(hook)
            hook_handles.append(handle)
            input_array = self._get_input_array_copy(input_array)
            self.model(input_array)

        detected = False
        for prev, current in zip(self.module_list[:-1], self.module_list[1:]):
            bias = getattr(prev, "bias", None)
            detected = (isinstance(current, self.normalization_layer)
                        and current.training  # TODO: do we want this check?
                        and isinstance(bias, torch.Tensor)
                        and bias.requires_grad)

            # if self.error:

                # if detected:
                #     rank_zero_warn(
                #
                #     )
        return not detected