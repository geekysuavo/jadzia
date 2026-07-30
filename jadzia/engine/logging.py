"""Logging functionality."""

# Standard
import logging
import threading
import time
from collections.abc import Callable
from typing import Any

# Third party
import jax


def _get_logger(level: int | str | None = None) -> logging.Logger:
    """Get the `jadzia` logger."""
    logger = logging.getLogger("jadzia")

    if not logger.handlers:
        logger.setLevel(level or logging.INFO)
        logger.propagate = False

        formatter = logging.Formatter(
            fmt="{asctime} {process} {levelname:>8s}: {message}",
            datefmt="%Y%m%d %H:%M:%S",
            style="{",
        )
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def _log_when_ready(
    func: Callable[[str], None],
    msg: str,
    kwargs: dict[str, Any],
    poll_interval: float = 0.1,
):
    """
    Start a thread that asynchronously copies data to the host and then logs it.

    Args:
        func: Logging function called once the data are ready.
        msg: Log message with keyword format strings.
        kwargs: Arguments for keyword format strings.
        poll_interval: Interval to wait (in seconds) for data to become ready.
    """

    def is_ready(leaf: Any) -> bool:
        return leaf.is_ready() if isinstance(leaf, jax.Array) else True

    def log_thread(func: Callable[[str], None], msg: str, kwargs: dict[str, Any]):
        while not all(jax.tree.map(is_ready, kwargs)):
            time.sleep(poll_interval)  # pragma: no cover
        func(msg.format(**kwargs))

    host_kwargs = jax.copy_to_host_async(kwargs)
    thread = threading.Thread(target=log_thread, args=(func, msg, host_kwargs))
    thread.start()


def debug(msg: str, *, jax_async: bool = False, **kwargs: Any):
    """Log a debug message."""
    if jax_async:
        _log_when_ready(debug, msg, kwargs)
    else:
        _get_logger().debug(msg.format(**kwargs))


def info(msg: str, *, jax_async: bool = False, **kwargs: Any):
    """Log an informational message."""
    if jax_async:
        _log_when_ready(info, msg, kwargs)
    else:
        _get_logger().info(msg.format(**kwargs))


def warning(msg: str, *, jax_async: bool = False, **kwargs: Any):
    """Log a warning message."""
    if jax_async:
        _log_when_ready(warning, msg, kwargs)
    else:
        _get_logger().warning(msg.format(**kwargs))


def error(msg: str, *, jax_async: bool = False, **kwargs: Any):
    """Log an error message."""
    if jax_async:
        _log_when_ready(error, msg, kwargs)
    else:
        _get_logger().error(msg.format(**kwargs))


__all__ = ["debug", "error", "info", "warning"]
