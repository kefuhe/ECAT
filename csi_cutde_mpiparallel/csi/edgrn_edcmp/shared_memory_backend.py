"""
Shared memory backend for EDCMP Green's functions.

This module enables multiple processes to share Green's function data
without reloading, providing significant performance improvements for
parallel computation.
"""

import os
import logging
import uuid
import numpy as np
from multiprocessing import shared_memory
from typing import Dict, Tuple, Optional, Any

logger = logging.getLogger(__name__)


class SharedGreenFunctions:
    """
    Manages Green's functions in shared memory for multi-process access.

    This class handles:
    - Loading Green's functions into shared memory (main process)
    - Accessing Green's functions from shared memory (worker processes)
    - Proper cleanup of shared memory resources

    Design:
    - Main process creates shared memory and loads data
    - Worker processes attach to existing shared memory by name
    - Shared memory is cleaned up when main process exits
    """

    def __init__(self):
        self.shm_blocks = {}  # name -> SharedMemory object
        self.metadata = {}    # Metadata for reconstructing arrays
        self.is_owner = False # Whether this instance created the shared memory

    def create_from_model(self, model, engine: str) -> Dict[str, Any]:
        """
        Create shared memory from a loaded EDCMP model.

        Parameters
        ----------
        model : object
            Loaded EDCMP model (ctypes)
        engine : str
            Engine type ('ctypes')

        Returns
        -------
        metadata : dict
            Metadata needed to reconstruct the model in worker processes
        """
        self.is_owner = True
        metadata = {'engine': engine, 'arrays': {}}

        # Extract Green's function arrays from model
        arrays_to_share = self._extract_arrays_from_model(model, engine)

        # Create shared memory for each array
        for name, arr in arrays_to_share.items():
            shm_name = f"edcmp_grn_{name}_{os.getpid()}_{uuid.uuid4().hex[:8]}"

            # Create shared memory block
            shm = shared_memory.SharedMemory(
                create=True,
                size=arr.nbytes,
                name=shm_name
            )

            # Copy array data to shared memory
            shm_arr = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
            shm_arr[:] = arr[:]

            # Store shared memory object and metadata
            self.shm_blocks[name] = shm
            metadata['arrays'][name] = {
                'shm_name': shm_name,
                'shape': arr.shape,
                'dtype': str(arr.dtype),
            }

            logger.debug(
                f"Created shared memory '{shm_name}': "
                f"shape={arr.shape}, dtype={arr.dtype}, size={arr.nbytes/1024/1024:.1f}MB"
            )

        # Store model metadata
        metadata['model_info'] = self._extract_model_info(model, engine)
        self.metadata = metadata

        logger.info(
            f"Created shared memory for {len(arrays_to_share)} arrays, "
            f"total size: {sum(arr.nbytes for arr in arrays_to_share.values())/1024/1024:.1f}MB"
        )

        return metadata

    def attach_from_metadata(self, metadata: Dict[str, Any]):
        """
        Attach to existing shared memory using metadata.

        Parameters
        ----------
        metadata : dict
            Metadata from create_from_model()
        """
        self.is_owner = False
        self.metadata = metadata

        # Attach to each shared memory block
        for name, arr_meta in metadata['arrays'].items():
            shm = shared_memory.SharedMemory(
                name=arr_meta['shm_name'],
                create=False
            )
            self.shm_blocks[name] = shm

        logger.debug(f"Attached to {len(self.shm_blocks)} shared memory blocks")

    def get_array(self, name: str) -> np.ndarray:
        """
        Get a numpy array view of shared memory.

        Parameters
        ----------
        name : str
            Array name

        Returns
        -------
        array : np.ndarray
            Read-only view of the shared array
        """
        if name not in self.shm_blocks:
            raise KeyError(f"Array '{name}' not found in shared memory")

        shm = self.shm_blocks[name]
        arr_meta = self.metadata['arrays'][name]

        # Create numpy array view
        arr = np.ndarray(
            arr_meta['shape'],
            dtype=np.dtype(arr_meta['dtype']),
            buffer=shm.buf
        )

        # Return read-only view to prevent accidental modification
        arr.flags.writeable = False
        return arr

    def reconstruct_model(self, module):
        """
        Reconstruct EDCMP model from shared memory.

        Parameters
        ----------
        module : module
            EDCMP module (edcmp4py_ctypes, etc.)

        Returns
        -------
        model : object
            Reconstructed EDCMP model with Green's functions
        """
        arrays = {name: self.get_array(name) for name in ('grnss', 'grnds', 'grncl') if name in self.shm_blocks}
        cls = module.EdcmpLayeredCtypes
        if hasattr(cls, 'from_shared_arrays'):
            model = cls.from_shared_arrays(
                grnss=arrays.get('grnss'),
                grnds=arrays.get('grnds'),
                grncl=arrays.get('grncl'),
                model_info=self.metadata['model_info'],
            )
        else:
            model = cls()
            model._grn_loaded = True
            for attr, val in self.metadata['model_info'].items():
                setattr(model, attr, val)
            for name, arr in arrays.items():
                setattr(model, name, arr)
        logger.debug(f"Reconstructed {self.metadata['engine']} model from shared memory")
        return model

    def cleanup(self):
        """Clean up shared memory resources."""
        if not self.is_owner:
            # Worker processes just close their handles
            for shm in self.shm_blocks.values():
                shm.close()
            logger.debug("Closed shared memory handles")
        else:
            # Owner process unlinks shared memory
            for name, shm in self.shm_blocks.items():
                shm.close()
                try:
                    shm.unlink()
                    logger.debug(f"Unlinked shared memory '{shm.name}'")
                except FileNotFoundError:
                    pass  # Already unlinked
            logger.info("Cleaned up shared memory")

        self.shm_blocks.clear()
        self.metadata.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False

    def _extract_arrays_from_model(self, model, engine: str) -> Dict[str, np.ndarray]:
        """Extract Green's function arrays from model."""
        arrays = {}

        # Ctypes engine stores arrays as attributes
        if hasattr(model, 'grnss') and model.grnss is not None:
            arrays['grnss'] = np.asarray(model.grnss)
        if hasattr(model, 'grnds') and model.grnds is not None:
            arrays['grnds'] = np.asarray(model.grnds)
        if hasattr(model, 'grncl') and model.grncl is not None:
            arrays['grncl'] = np.asarray(model.grncl)

        return arrays

    def _extract_model_info(self, model, engine: str) -> Dict[str, Any]:
        """Extract model metadata."""
        info = {}

        # Extract grid parameters
        for attr in ['nr', 'nz', 'r1', 'r2', 'z1', 'z2', 'zrec0', 'lam', 'mu']:
            if hasattr(model, attr):
                info[attr] = getattr(model, attr)

        return info


def create_shared_greens(engine: str, grn_dir: str, workdir: str = ".",
                        module_dir: Optional[str] = None) -> Tuple[SharedGreenFunctions, Dict]:
    """
    Create shared memory Green's functions (main process).

    Parameters
    ----------
    engine : str
        Engine type ('ctypes')
    grn_dir : str
        Green's function directory
    workdir : str
        Working directory
    module_dir : str, optional
        Module directory

    Returns
    -------
    shared_grn : SharedGreenFunctions
        Shared memory manager
    metadata : dict
        Metadata for worker processes
    """
    from .edcmp_backends import _load_inmemory_backend

    # Load model normally (this will use cache)
    model = _load_inmemory_backend(
        engine,
        grn_dir=grn_dir,
        workdir=workdir,
        module_dir=module_dir
    )

    # Create shared memory
    shared_grn = SharedGreenFunctions()
    metadata = shared_grn.create_from_model(model, engine)

    return shared_grn, metadata


def attach_shared_greens(metadata: Dict, module_dir: Optional[str] = None):
    """
    Attach to shared memory Green's functions (worker process).

    Parameters
    ----------
    metadata : dict
        Metadata from create_shared_greens()
    module_dir : str, optional
        Module directory

    Returns
    -------
    model : object
        EDCMP model with Green's functions from shared memory
    """
    from .edcmp_backends import _import_edcmp4py_module

    engine = metadata['engine']

    # Import module
    module = _import_edcmp4py_module(engine, module_dir=module_dir)

    # Attach to shared memory
    shared_grn = SharedGreenFunctions()
    shared_grn.attach_from_metadata(metadata)

    # Reconstruct model
    model = shared_grn.reconstruct_model(module)

    return model
