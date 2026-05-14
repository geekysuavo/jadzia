"""Tests of shared memory data frames."""

# Standard
import os
import random
import time
from multiprocessing import Process
from multiprocessing.shared_memory import SharedMemory

# Third party
import polars as pl
import pytest

# Dax
from jadzia.data.sharedmemory import SharedDataFrame

_SHM_NAME: str = f"sdftest{os.getpid()}"


@pytest.mark.parametrize("name", [None, _SHM_NAME])
def test_shared_data_frame(name):
    """Shared data frames created by one process can be read from another."""
    num_rows = 1000
    value = [random.random() for _ in range(num_rows)]
    other = random.choices(["foo", "bar", "hello"], k=num_rows)
    df = pl.DataFrame({"value": value, "other": other})

    shm_a = SharedDataFrame(df, name=name, create=True)
    shm_b = SharedDataFrame(df, name=name, create=False)

    rows_a = list(shm_a)
    rows_b = list(shm_b)
    assert rows_a == rows_b
    assert rows_a == df.to_dicts()

    row_iter = iter(shm_a)
    for index in range(5):
        assert next(row_iter) == rows_a[index]

    rows_c = list(row_iter)
    assert rows_c == rows_a

    for index in range(5):
        assert shm_a[-index] == rows_a[-index]

    with pytest.raises(IndexError, match="out of range"):
        shm_a[num_rows]

    with pytest.raises(IndexError, match="out of range"):
        shm_a[-num_rows - 1]


def _creator_process(name):
    """Process that mimics a very slow shared data frame creation."""
    time.sleep(0.5)
    shm = SharedMemory(name, create=True, size=8)
    time.sleep(0.5)
    shm.buf[0] = 0xff
    time.sleep(0.5)
    shm.close()


def test_wait_ready(mocker):
    mocker.patch("jadzia.data.sharedmemory._WAIT_INTERVAL", 0.1)

    proc = Process(target=_creator_process, args=(_SHM_NAME,))
    proc.start()

    df = pl.DataFrame({"x": [1, 2, 3]})
    SharedDataFrame(df, name=_SHM_NAME, create=False)

    proc.join()
