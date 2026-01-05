import logging
import sys
from typing import Optional
from mpi4py import MPI
import logging

def _get_mpi_rank():
    return MPI.COMM_WORLD.Get_rank()

def _create_formatter(rank):
    return logging.Formatter(f'%(asctime)s - [Rank {rank}] - %(name)s - %(message)s')

# --- Fallback function for internal library use ---
def ensure_default_logging(verbose: bool = True):
    """
    Internal use: If no logging is configured by the user, automatically set up a simple console output.
    Prevents 'No handlers could be found for logger' warnings or silent failures.
    """
    # 1. Core defense: If a Handler already exists, do not interfere to prevent duplicate logs!
    if logging.getLogger().hasHandlers():
        return

    # 2. If verbose=False and no logging is configured, do not disturb (or just add a NullHandler)
    if not verbose:
        logging.getLogger().addHandler(logging.NullHandler())
        return

    # 3. Only configure default behavior if user hasn't configured and verbose=True
    rank = _get_mpi_rank()
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_create_formatter(rank))

    if rank == 0:
        handler.setLevel(logging.INFO)
    else:
        handler.setLevel(logging.ERROR)

    root_logger.addHandler(handler)
    # Only prompt on Rank 0
    if rank == 0:
        logging.getLogger(__name__).info("Smart Logging: Auto-configured default console logging.")

# --- Main function for user configuration ---
def setup_parallel_logging(
    log_filename: Optional[str] = "run.log",
    level: int = logging.INFO,
    file_mode: str = 'w',
    log_format: Optional[str] = None
) -> logging.Logger:
    """
    Configure a logging system suitable for MPI parallel environments.

    Principles:
    - Rank 0: Print all INFO and above logs to both console and file.
    - Rank > 0: Only print ERROR and above logs to the console, do not write to file.

    Args:
        log_filename (str): Log file path. If None, do not write to file.
        level (int): Global logging level (default logging.INFO).
        file_mode (str): File write mode, 'w' for overwrite or 'a' for append.
        log_format (str): Custom log format. If None, use the default format with Rank info.

    Returns:
        logging.Logger: Configured root logger.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    # Get the root logger
    root_logger = logging.getLogger()

    # 0. Prevent duplicate configuration (if handlers exist, already configured)
    if root_logger.hasHandlers():
        return root_logger

    root_logger.setLevel(level)

    # 1. Define format (automatically inject Rank info)
    if log_format is None:
        log_format = f'%(asctime)s - [Rank {rank}] - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(log_format)

    # 2. Configure console output (StreamHandler)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # Key trick: set console silence level by Rank
    if rank == 0:
        console_handler.setLevel(level)  # Main process: normal output
    else:
        console_handler.setLevel(logging.ERROR) # Worker process: only serious errors

    root_logger.addHandler(console_handler)

    # 3. Configure file output (FileHandler) - only for Rank 0
    if rank == 0 and log_filename:
        try:
            file_handler = logging.FileHandler(log_filename, mode=file_mode, encoding='utf-8')
            file_handler.setFormatter(formatter)
            file_handler.setLevel(level)
            root_logger.addHandler(file_handler)
        except Exception as e:
            # If file cannot be opened, at least report error to console
            console_handler.setLevel(logging.INFO) # Temporarily raise level to ensure error is visible
            root_logger.error(f"Failed to set up log file: {e}")

    # 4. Suppress verbose logs from third-party libraries (optional, recommended)
    # For example, matplotlib is very noisy in debug mode
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)

    return root_logger