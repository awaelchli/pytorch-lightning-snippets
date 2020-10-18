from unittest import mock

import pytest

from monitor import TrainingDataMonitor
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import LoggerCollection, TensorBoardLogger
from tests.templates.base import TemplateModelBase


@pytest.mark.parametrize(
    ["log_every_n_steps", "max_steps", "expected_calls"], [pytest.param(3, 10, 3)]
)
def test_row_log_interval_override(
    tmpdir, log_every_n_steps, max_steps, expected_calls
):
    """ Test logging interval set by row_log_interval argument. """
    monitor = TrainingDataMonitor(log_every_n_steps=log_every_n_steps)
    model = TemplateModelBase()
    trainer = Trainer(
        default_root_dir=tmpdir,
        log_every_n_steps=1,
        max_steps=max_steps,
        callbacks=[monitor],
    )
    with mock.patch.object(
        TrainingDataMonitor, "log_histogram", wraps=monitor.log_histogram
    ) as mocked_monitor:
        trainer.fit(model)
        assert mocked_monitor.call_count == (
            expected_calls * 2
        )  # 2 tensors per log call


@pytest.mark.parametrize(
    ["log_every_n_steps", "max_steps", "expected_calls"],
    [
        pytest.param(1, 5, 5),
        pytest.param(2, 5, 2),
        pytest.param(5, 5, 1),
        pytest.param(6, 5, 0),
    ],
)
def test_row_log_interval_fallback(
    tmpdir, log_every_n_steps, max_steps, expected_calls
):
    """ Test that if row_log_interval not set in the callback, fallback to what is defined in the Trainer. """
    monitor = TrainingDataMonitor()
    model = TemplateModelBase()
    trainer = Trainer(
        default_root_dir=tmpdir,
        log_every_n_steps=log_every_n_steps,
        max_steps=max_steps,
        callbacks=[monitor],
    )
    with mock.patch.object(
        TrainingDataMonitor, "log_histogram", wraps=monitor.log_histogram
    ) as mocked_monitor:
        trainer.fit(model)
        assert mocked_monitor.call_count == (
            expected_calls * 2
        )  # 2 tensors per log call


def test_no_logger_warning():
    monitor = TrainingDataMonitor()
    trainer = Trainer(logger=False, callbacks=[monitor])
    with pytest.warns(
        UserWarning, match="Cannot log histograms because Trainer has no logger"
    ):
        monitor.on_train_start(trainer, pl_module=None)


def test_unsupported_logger_warning(tmpdir):
    monitor = TrainingDataMonitor()
    trainer = Trainer(
        logger=LoggerCollection([TensorBoardLogger(tmpdir)]), callbacks=[monitor]
    )
    with pytest.warns(
        UserWarning, match="does not support logging with LoggerCollection"
    ):
        print(trainer.log_every_n_steps)
        monitor.on_train_start(trainer, pl_module=None)
