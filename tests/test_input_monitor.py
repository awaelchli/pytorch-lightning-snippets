from unittest import mock

import pytest

from monitor.input_monitor import InputMonitor
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import LoggerCollection, TensorBoardLogger
from tests.templates.base import TemplateModelBase


@pytest.mark.parametrize(["row_log_interval", "max_steps", "expected_calls"], [
    pytest.param(3, 10, 3)
])
def test_row_log_interval_override(tmpdir, row_log_interval, max_steps, expected_calls):
    """ Test logging interval set by row_log_interval argument. """
    monitor = InputMonitor(row_log_interval=row_log_interval)
    model = TemplateModelBase()
    trainer = Trainer(
        default_root_dir=tmpdir,
        row_log_interval=1,
        max_steps=max_steps,
        callbacks=[monitor]
    )
    with mock.patch.object(InputMonitor, 'log_histograms', wraps=monitor.log_histograms) as mocked_monitor:
        trainer.fit(model)
        assert mocked_monitor.call_count == expected_calls


@pytest.mark.parametrize(["row_log_interval", "max_steps", "expected_calls"], [
    pytest.param(1, 5, 5),
    pytest.param(2, 5, 2),
    pytest.param(5, 5, 1),
    pytest.param(6, 5, 0)
])
def test_row_log_interval_fallback(tmpdir, row_log_interval, max_steps, expected_calls):
    """ Test that if row_log_interval not set in the callback, fallback to what is defined in the Trainer. """
    monitor = InputMonitor()
    model = TemplateModelBase()
    trainer = Trainer(
        default_root_dir=tmpdir,
        row_log_interval=row_log_interval,
        max_steps=max_steps,
        callbacks=[monitor]
    )
    with mock.patch.object(InputMonitor, 'log_histograms', wraps=monitor.log_histograms) as mocked_monitor:
        trainer.fit(model)
        assert mocked_monitor.call_count == expected_calls


def test_no_logger_warning():
    monitor = InputMonitor()
    trainer = Trainer(logger=False, callbacks=[monitor])
    with pytest.warns(UserWarning, match="Cannot log histograms because Trainer has no logger"):
        monitor.on_train_start(trainer, pl_module=None)


def test_unsupported_logger_warning(tmpdir):
    monitor = InputMonitor()
    trainer = Trainer(logger=LoggerCollection([TensorBoardLogger(tmpdir)]), callbacks=[monitor])
    with pytest.warns(UserWarning, match="does not support logging with LoggerCollection"):
        monitor.on_train_start(trainer, pl_module=None)