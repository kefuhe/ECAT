import logging
import sys
from typing import Optional, Dict, Tuple
from mpi4py import MPI

# Default application namespaces whose INFO/DEBUG messages are allowed through.
# Third-party libraries inherit the root logger's WARNING level and are
# therefore silenced automatically — no blacklist maintenance needed.
_DEFAULT_APP_NAMESPACES = ('eqtools', 'csi')


def _get_mpi_rank():
    return MPI.COMM_WORLD.Get_rank()


def _create_formatter(rank, app_name=None):
    prefix = f"[{app_name}] " if app_name else ""
    return logging.Formatter(
        f'{prefix}%(asctime)s - [Rank {rank}] - %(name)s - %(levelname)s - %(message)s'
    )


def _configure_levels(root_logger, level, app_namespaces, third_party_levels):
    """Configure logger levels using a whitelist strategy.

    - Root logger is always set to WARNING so that third-party INFO/DEBUG
      messages are suppressed by default (no blacklist needed).
    - Only the app namespaces (eqtools, csi, …) are lowered to *level*.
    - *third_party_levels* can temporarily **un-mute** a specific library
      for debugging (e.g. ``{'fontTools': logging.DEBUG}``).
    """
    # Root at WARNING — all third-party INFO is silenced by inheritance
    root_logger.setLevel(logging.WARNING)

    # Whitelist: only our namespaces get the user-requested level
    for ns in app_namespaces:
        logging.getLogger(ns).setLevel(level)

    # Optional: override specific third-party loggers (for debugging)
    if third_party_levels:
        for name, lvl in third_party_levels.items():
            logging.getLogger(name).setLevel(lvl)


# --- Fallback function for internal library use ---
def ensure_default_logging(verbose: bool = True):
    """
    Internal use: If no logging is configured by the user, automatically set up a simple console output.
    Prevents 'No handlers could be found for logger' warnings or silent failures.
    """
    root_logger = logging.getLogger()

    # Always apply level configuration (even if handlers already exist,
    # e.g. when Jupyter/IPython pre-installs a handler).
    _configure_levels(root_logger, logging.INFO, _DEFAULT_APP_NAMESPACES, None)

    # If handlers already exist, don't add more (prevent duplicates).
    if root_logger.hasHandlers():
        return

    if not verbose:
        root_logger.addHandler(logging.NullHandler())
        return

    rank = _get_mpi_rank()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_create_formatter(rank))

    if rank == 0:
        handler.setLevel(logging.INFO)
    else:
        handler.setLevel(logging.ERROR)

    root_logger.addHandler(handler)
    if rank == 0:
        logging.getLogger(__name__).info("Smart Logging: Auto-configured default console logging.")


# --- Main function for user configuration ---
def setup_parallel_logging(
    log_filename: Optional[str] = "run.log",
    level: int = logging.INFO,
    file_mode: str = 'w',
    log_format: Optional[str] = None,
    console_output: bool = True,
    third_party_levels: Optional[Dict[str, int]] = None,
    app_name: Optional[str] = 'ECAT',
    app_namespaces: Tuple[str, ...] = _DEFAULT_APP_NAMESPACES,
) -> logging.Logger:
    """
    Configure a logging system suitable for MPI parallel environments.

    Uses a **whitelist strategy**: the root logger is set to WARNING so that
    all third-party libraries are silenced by default.  Only the namespaces
    listed in *app_namespaces* are lowered to *level* (INFO or DEBUG).

    Principles:
    - Rank 0: Print all app-level logs to both console and file.
    - Rank > 0: Only print ERROR and above logs to the console, do not write to file.

    Args:
        log_filename (str): Log file path. If None, do not write to file.
        level (int): Logging level for app namespaces (default logging.INFO).
        file_mode (str): File write mode, 'w' for overwrite or 'a' for append.
        log_format (str): Custom log format. If None, use the default format with Rank info.
        console_output (bool): Whether to print logs to console on Rank 0 (default True).
            If False, only WARNING/ERROR will be printed.
        third_party_levels (dict): Dictionary of {logger_name: level} to **un-mute**
            specific third-party libraries for debugging.
        app_name (str): Optional application name to prefix in logs (e.g., "ECAT").
        app_namespaces (tuple): Logger name prefixes whose messages are shown at
            *level*. Default: ``('eqtools', 'csi')``.

    Returns:
        logging.Logger: Configured root logger.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    root_logger = logging.getLogger()

    # Always apply level configuration — even if handlers already exist
    # (e.g. Jupyter pre-installs a handler, or logging.basicConfig was
    # called elsewhere).  This guarantees third-party noise is suppressed.
    _configure_levels(root_logger, level, app_namespaces, third_party_levels)

    # Don't add duplicate handlers if already configured.
    if root_logger.hasHandlers():
        return root_logger

    # 1. Define format (automatically inject Rank info)
    if log_format is None:
        log_format = _create_formatter(rank, app_name)._fmt
    formatter = logging.Formatter(log_format)

    # 2. Configure console output (StreamHandler)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    if rank == 0:
        if console_output:
            console_handler.setLevel(level)
        else:
            console_handler.setLevel(logging.WARNING)
    else:
        console_handler.setLevel(logging.ERROR)

    root_logger.addHandler(console_handler)

    # 3. Configure file output (FileHandler) - only for Rank 0
    if rank == 0 and log_filename:
        try:
            file_handler = logging.FileHandler(log_filename, mode=file_mode, encoding='utf-8')
            file_handler.setFormatter(formatter)
            file_handler.setLevel(level)
            root_logger.addHandler(file_handler)
        except Exception as e:
            console_handler.setLevel(logging.INFO)
            root_logger.error(f"Failed to set up log file: {e}")

    return root_logger