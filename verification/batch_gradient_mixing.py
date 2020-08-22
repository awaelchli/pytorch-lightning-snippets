from typing import Callable

from pytorch_lightning import Callback
from verification.base import ModelVerificationBase


class BatchMixingVerification(ModelVerificationBase):

    def warning_message(self, *args, **kwargs):
        message = (
            "Your model is mixing data across the batch dimension."
            " This can lead to wrong gradient updates in the optimizer."
            " Check the operations that reshape and permute tensor dimensions in your model."
        )
        return message

    def check(self, input_array=None, input_mapping: Callable = None, output_mapping: Callable = None, sample_idx=0):
        input_mapping = input_mapping or (lambda x: x)
        output_mapping = output_mapping or (lambda x: x)
        input_array = self._get_input_array_copy(input_array)
        input_batch = input_mapping(input_array)
        assert input_batch.size(0) > 1, "batch_size must be greater than 1 for this test"

        input_batch.requires_grad = True
        self.model.zero_grad()
        output = self.model(input_array)

        # backward on the i-th sample should lead to gradient only in i-th input slice
        output_mapping(output)[sample_idx].sum().backward()

        zero_grad_inds = list(range(len(input_batch)))
        zero_grad_inds.pop(sample_idx)

        has_grad_outside_sample = input_batch.grad[zero_grad_inds].abs().sum().item() > 0
        has_grad_in_sample = input_batch.grad[sample_idx].abs().sum().item() > 0
        return has_grad_in_sample and not has_grad_outside_sample


class BatchMixingVerificationCallback(Callback):

    def __init__(self, input_mapping: Callable = None, output_mapping: Callable = None, sample_idx=0):
        self._input_mapping = input_mapping
        self._output_mapping = output_mapping
        self._sample_idx = sample_idx

    def on_train_start(self, trainer, pl_module):
        verification = BatchMixingVerification(pl_module)
        result = verification.check(
            input_array=pl_module.example_input_array,
            input_mapping=self._input_mapping,
            output_mapping=self._output_mapping,
            sample_idx=self._sample_idx,
        )
        # if not result:
        #     rank_zero_warn(
        #
        #     )