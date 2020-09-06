from typing import List, Optional

from monitor.data_monitor_base import DataMonitorBase
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.utilities import rank_zero_warn
import torch.nn as nn


class ModuleDataMonitor(DataMonitorBase):

    def __init__(self, submodules: Optional[List[str]] = None, row_log_interval: int = None):
        super().__init__(row_log_interval=row_log_interval)
        self._submodule_names = submodules
        self._hook_handles = []

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule):
        super().on_train_start(trainer, pl_module)
        submodule_dict = dict(pl_module.named_modules())

        if self._submodule_names is None:
            handle = self.register_hook("root", pl_module)
            self._hook_handles = [handle]
        else:
            self._hook_handles = []
            for name in self._submodule_names:
                handle = self.register_hook(name, submodule_dict[name])
                self._hook_handles.append(handle)

    def on_train_end(self, trainer, pl_module):
        for handle in self._hook_handles:
            handle.remove()

    def register_hook(self, module_name: str, module: nn.Module):

        def hook(_, inp, out):
            self.log_histograms(inp, group=f"input/{module_name}")
            self.log_histograms(out, group=f"output/{module_name}")

        handle = module.register_forward_hook(hook)
        return handle
