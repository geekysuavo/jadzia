"""Tests of logging."""

# Standard
import logging
import time

# Third party
import jax
import pytest

# Dax
import jadzia.engine.logging as log


@pytest.mark.parametrize("jax_async", [False, True])
@pytest.mark.parametrize(
    "func",
    [log.debug, log.info, log.warning, log.error],
    ids=["debug", "info", "warning", "error"],
)
def test_log(caplog, func, jax_async):
    """Messages may be logged to `stderr` at varying levels, default is `INFO`."""
    msg = f"hello {func.__name__} {{data}}"
    data = jax.numpy.array(1.5)

    with caplog.at_level(logging.DEBUG, logger="jadzia"):
        func(msg, data=data, jax_async=jax_async)
        if jax_async:
            time.sleep(0.05)

        expected = f"hello {func.__name__} 1.5"
        assert expected in caplog.text

    caplog.clear()
