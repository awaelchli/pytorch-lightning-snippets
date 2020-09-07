from typing import List, Optional, Union

import torch.nn as nn

from monitor.data_monitor_base import DataMonitorBase
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.utilities import rank_zero_warn


class ModuleDataMonitor(DataMonitorBase):

    GROUP_NAME_INPUT = "input"
    GROUP_NAME_OUTPUT = "output"

    def __init__(self, submodules: Optional[Union[bool, List[str]]] = None, row_log_interval: int = None):
        """
        Args:
            submodules: If `True`, logs the in- and output histograms of every submodule in the
                LightningModule, including the root module itself.
                This parameter can also take a list of names of specifc submodules (see example below).
                Default: `None`, logs only the in- and output of the root module.
            row_log_interval: The interval at which histograms should be logged. This defaults to the
                interval defined in the Trainer. Use this to override the Trainer default.

        Note:
            A too low value for `row_log_interval` may have a significant performance impact
            especially when many submodules are involved, since the logging occurs during the forward pass.
            It should only be used for debugging purposes.

        Example:

            .. code-block:: python

                # log the in- and output histograms of the `forward` in LightningModule
                trainer = Trainer(callbacks=[ModuleDataMonitor()])

                # all submodules in LightningModule
                trainer = Trainer(callbacks=[ModuleDataMonitor(submodules=True)])

                # specific submodules
                trainer = Trainer(callbacks=[ModuleDataMonitor(submodules=["generator", "generator.conv1"])])

        """
        super().__init__(row_log_interval=row_log_interval)
        self._submodule_names = submodules
        self._hook_handles = []

    def submodule_names(self, root_module: nn.Module):
        # default is the root module only
        names = [""]

        if isinstance(self._submodule_names, list):
            names = self._submodule_names

        if self._submodule_names is True:
            names = [name for name, _ in root_module.named_modules()]

        return names

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule):
        super().on_train_start(trainer, pl_module)
        submodule_dict = dict(pl_module.named_modules())
        self._hook_handles = []
        for name in self.submodule_names(pl_module):
            if name not in submodule_dict:
                rank_zero_warn(
                    f"{name} is not a valid identifier for a submodule in {pl_module.__class__.__name__},"
                    " skipping this key."
                )
                continue
            handle = self.register_hook(name, submodule_dict[name])
            self._hook_handles.append(handle)

    def on_train_end(self, trainer, pl_module):
        for handle in self._hook_handles:
            handle.remove()

    def register_hook(self, module_name: str, module: nn.Module):
        input_group_name = f"{self.GROUP_NAME_INPUT}/{module_name}" if module_name else self.GROUP_NAME_INPUT
        output_group_name = f"{self.GROUP_NAME_OUTPUT}/{module_name}" if module_name else self.GROUP_NAME_OUTPUT

        def hook(_, inp, out):
            inp = inp[0] if len(inp) == 1 else inp
            self.log_histograms(inp, group=input_group_name)
            self.log_histograms(out, group=output_group_name)

        handle = module.register_forward_hook(hook)
        return handle
