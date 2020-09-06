from monitor.data_monitor_base import DataMonitorBase


class TrainingDataMonitor(DataMonitorBase):

    def __init__(self, row_log_interval: int = None):
        """
        Callback that logs the histogram of values in the batched data passed to `training_step`.

        Args:
            row_log_interval: The interval at which histograms should be logged. This defaults to the
                interval defined in the Trainer. Use this to override the Trainer default.
        """
        super().__init__(row_log_interval=row_log_interval)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        super().on_train_batch_start(trainer, pl_module, batch, batch_idx, dataloader_idx)
        self.log_histograms(batch, group="training_step")