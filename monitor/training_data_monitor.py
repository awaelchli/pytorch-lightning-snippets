from monitor.data_monitor_base import DataMonitorBase


class TrainingDataMonitor(DataMonitorBase):

    GROUP_NAME = "training_step"

    def __init__(self, row_log_interval: int = None):
        """
        Callback that logs the histogram of values in the batched data passed to `training_step`.

        Args:
            row_log_interval: The interval at which histograms should be logged. This defaults to the
                interval defined in the Trainer. Use this to override the Trainer default.

        Example:

            .. code-block:: python

                # log histogram of training data passed to `LightningModule.training_step`
                trainer = Trainer(callbacks=[TrainingDataMonitor()])
        """
        super().__init__(row_log_interval=row_log_interval)

    def on_train_batch_start(self, trainer, pl_module, batch, *args, **kwargs):
        super().on_train_batch_start(trainer, pl_module, batch, *args, **kwargs)
        self.log_histograms(batch, group=self.GROUP_NAME)
