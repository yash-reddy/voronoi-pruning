"""Helper functions commonly used by other scripts in the directory."""

import re

MAX_SCORE = int(1e8)
# Depends on the VRAM, this should work for VRAM >=32G
MAX_BATCH_SIZE = 5000
VOLUME_METRIC = "volume"
MEAN_ERROR_METRIC = "mean_error"
MAX_ERROR_METRIC = "max_error"
PRUNE_ORDER_BASENAME = "pruning_orders"


def log_args(args, logger):
    """Logs the arguments used to run the script."""
    logger.info("Running the script with the following Arguments:")
    for arg, value in vars(args).items():
        logger.info("%20s: %s", arg, value)
    logger.info("******** End of Arguments ********")


def prune_order_filename_mods(non_iterative, step_size, chunk_idx=None, beam_size=1):
    """Generates a string of mode markers to be appended to a filename based on pruning configuration."""
    mode_markers = ""
    if non_iterative:
        mode_markers += ".non_iterative"
    if chunk_idx is not None:
        mode_markers += f".c_{chunk_idx}"
    if step_size != 1:
        mode_markers += f".step_{step_size}"
    if beam_size != 1:
        mode_markers += f".beam_{beam_size}"
    return mode_markers


def get_prune_order_files(index_dir, non_iterative, step_size, beam_size=1):
    """Retrieves prune order files from the specified index directory based on the pruning configuration."""
    mode_markers = ""
    if non_iterative:
        mode_markers += ".non_iterative"
    mode_markers += r"(|\.c_[\d]+)"
    if step_size != 1:
        mode_markers += f".step_{step_size}"
    if beam_size != 1:
        mode_markers += f".beam_{beam_size}"
    order_file_regex = re.compile(rf"{PRUNE_ORDER_BASENAME}{mode_markers}.npy")
    return sorted([f for f in index_dir.iterdir() if order_file_regex.match(f.name)])
