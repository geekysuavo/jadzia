"""Tests of logging."""

# Standard
import time

# Third party
import jax
import pytest

# Dax
import jadzia.engine.logging as logging


@pytest.mark.parametrize("jax_async", [False, True])
@pytest.mark.parametrize(
    "func",
    [logging.debug, logging.info, logging.warning, logging.error],
    ids=["debug", "info", "warning", "error"],
)
def test_debug(capsys, func, jax_async):
    """Messages may be logged to `stderr` at varying levels, default is `INFO`."""
    msg = "hello {data}"
    data = jax.numpy.array(1.5)

    func(msg, data=data, jax_async=jax_async)
    if jax_async:
        time.sleep(0.05)

    captured = capsys.readouterr()
    if func is logging.debug:
        assert captured.err == ""
    else:
        assert "hello 1.5" in captured.err
