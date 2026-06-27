"""The jadzia machine learning framework."""

# Third party
from absl.flags import FLAGS
from absl.logging import set_verbosity

# required to support multiprocessing data loaders.
FLAGS([__file__])
del FLAGS

# suppress innocuous warnings during checkpointing.
set_verbosity("ERROR")
del set_verbosity
