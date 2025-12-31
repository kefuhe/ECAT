import functools
import inspect
from typing import Dict, Any, Optional

# Import geometry base classes
# Ensure these files exist in the same directory
from .AdaptiveTriangularPatches import AdaptiveTriangularPatches
from .AdaptiveRectangularPatches import AdaptiveRectangularPatches

# =============================================================================
# 1. Registry System (For Help & Documentation & Config Validation)
# =============================================================================
class PerturbationRegistry:
    """
    Global Registry: Stores the method names, descriptions, and parameters 
    for all perturbation methods across different fault classes.
    """
    _methods = {}

    @classmethod
    def register(cls, class_name, method_name, meta_info):
        """Register a method with its metadata under a specific class name."""
        if class_name not in cls._methods:
            cls._methods[class_name] = {}
        cls._methods[class_name][method_name] = meta_info

    @classmethod
    def get_help(cls, instance_or_name=None):
        """
        Retrieve the help dictionary.
        
        Args:
            instance_or_name: Can be an instance object, a class object, or a string class name.
            
        Returns:
            dict: A dictionary of available methods and their metadata.
            If an instance/class is provided, it returns methods from the entire 
            inheritance chain (MRO).
        """
        # If no argument, return the entire registry
        if instance_or_name is None:
            return cls._methods

        # Handling String Input (e.g., from Config file)
        # Returns exactly what is registered under this string name.
        if isinstance(instance_or_name, str):
            return cls._methods.get(instance_or_name, {})

        # Handling Object/Class Input (e.g., fault.help())
        # Uses MRO to collect methods from all parent classes.
        target_cls = instance_or_name if isinstance(instance_or_name, type) else instance_or_name.__class__
        
        combined_methods = {}
        # Iterate MRO in reverse (Parent -> Child) so children can overwrite parents
        for base in reversed(inspect.getmro(target_cls)):
            base_name = base.__name__
            if base_name in cls._methods:
                combined_methods.update(cls._methods[base_name])
                
        return combined_methods

# =============================================================================
# 2. Enhanced Decorator (Tracks State & Records Documentation)
# =============================================================================
def track_mesh_update(update_mesh=False, update_laplacian=False, update_area=False, 
                      description="", params_info=None, expected_perturbations_count=None):
    """
    Decorator: Tracks the update state of the mesh/laplacian while recording 
    documentation information for the registry.

    Args:
        update_mesh (bool): If True, sets self.mesh_updated = True after execution.
        update_laplacian (bool): If True, sets self.laplacian_updated = True.
        update_area (bool): If True, sets self.area_updated = True.
        description (str): A brief description of what the perturbation does.
        params_info (dict/str): Information about expected parameters.
        expected_perturbations_count (int, optional): Validates the length of the 'perturbations' array.
    """
    def decorator(func):
        # Mount metadata for the Metaclass/Registry to read
        func._bayesian_meta = {
            "description": description,
            "params": params_info,
            "flags": {"mesh": update_mesh, "lap": update_laplacian}
        }
        
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            perturbations = args[0] if args else kwargs.get('perturbations', None)
            if perturbations is None:
                raise ValueError(f"Method '{func.__name__}': The 'perturbations' parameter is required.")
            
            # Validate parameter count if specified
            if expected_perturbations_count is not None and len(perturbations) != expected_perturbations_count:
                raise ValueError(f"Method '{func.__name__}': Expected {expected_perturbations_count} elements in 'perturbations', but got {len(perturbations)}.")

            # Reset flags before execution
            self.mesh_updated = False
            self.laplacian_updated = False
            self.area_updated = False
            
            # Execute the actual perturbation function
            result = func(self, *args, **kwargs)
            
            # Update flags based on decorator arguments
            if update_mesh: self.mesh_updated = True
            if update_laplacian: self.laplacian_updated = True
            if update_area: self.area_updated = True
            
            return result
        
        wrapper._is_decorated = True
        # Ensure metadata is accessible via the wrapper
        wrapper._bayesian_meta = func._bayesian_meta
        return wrapper
    return decorator

# =============================================================================
# 3. Metaclass (Auto-Registration & Validation)
# =============================================================================
class PerturbationMeta(type):
    """
    Metaclass: Automatically categorizes perturbation methods, enforces the use 
    of the decorator, and registers methods to the PerturbationRegistry.
    """
    def __new__(mcs, name, bases, attrs):
        attrs['perturbation_methods'] = {}
        attrs['bayesian_perturbation_methods'] = {}
        
        for key, value in attrs.items():
            if callable(value) and key.startswith('perturb_'):
                # Heuristic to identify Bayesian methods: (self, perturbations, ...)
                try:
                    parameters = list(inspect.signature(value).parameters.values())
                    is_bayesian = (len(parameters) >= 2 and 
                                   parameters[0].name == 'self' and 
                                   parameters[1].name == 'perturbations')
                except ValueError:
                    is_bayesian = False

                if is_bayesian:
                    attrs['bayesian_perturbation_methods'][key] = attrs[key]
                    
                    # 1. Enforce Decorator Usage
                    if not getattr(value, '_is_decorated', False):
                        raise TypeError(f"The method '{key}' in class '{name}' is not decorated with '@track_mesh_update'.")
                    
                    # 2. Register to Global Registry
                    meta_info = getattr(value, '_bayesian_meta', {"description": "No description", "params": "N/A"})
                    PerturbationRegistry.register(name, key, meta_info)
                    
                else:
                    # General perturbation methods
                    attrs['perturbation_methods'][key] = attrs[key]
                    
        return super().__new__(mcs, name, bases, attrs)

# =============================================================================
# 4. Perturbation Base Logic
# =============================================================================
class PerturbationBase(metaclass=PerturbationMeta):
    """
    Base class providing logic for state flags and dynamic method dispatch.
    """
    def __init__(self):
        self.geometry_updated = False
        self.mesh_updated = False
        self.laplacian_updated = False
        self.area_updated = False

    def is_mesh_updated(self): return self.mesh_updated
    def is_laplacian_updated(self): return self.laplacian_updated

    def perturb(self, method, **kwargs):
        """
        Dynamic dispatcher for perturbation methods.
        """
        if not method.startswith('perturb_'):
            raise ValueError("The method name must start with 'perturb_'")
        if 'perturbations' not in kwargs:
            raise ValueError("The 'perturbations' argument is required")

        # Search in general methods first
        perturb_method = self.perturbation_methods.get(method, None)
        
        # Fallback: Search in Bayesian methods
        if perturb_method is None:
             perturb_method = getattr(self, method, None)

        if perturb_method is None:
            # Gather available methods for error message
            available = list(self.perturbation_methods.keys()) + list(getattr(self, 'bayesian_perturbation_methods', {}).keys())
            raise ValueError(f"Method '{method}' not found. Available methods: {available}")

        return perturb_method(**kwargs)

# =============================================================================
# 5. Shared Memory Mechanism (MPI Support)
# =============================================================================
SHARED_ATTRIBUTES = [
    'xf', 'yf', 'lon', 'lat', 'xi', 'yi', 'loni', 'lati', 'top', 'depth', 'z_patches',
    'top_coords', 'top_coords_ll', 'layers', 'layers_ll',
    'bottom_coords', 'bottom_coords_ll', 'Faces', 'Vertices',
    'Vertices_ll', 'patch', 'patchll', 'area', 'GL'
]

class SharedFaultInfo:
    def __init__(self):
        for attr in SHARED_ATTRIBUTES:
            setattr(self, f"_{attr}", None)

def shared_property(attr_name):
    def getter(self): return getattr(self.shared_info, f"_{attr_name}")
    def setter(self, value): setattr(self.shared_info, f"_{attr_name}", value)
    return property(getter, setter)

# =============================================================================
# 6. The Core Base Class: BayesianTriFaultBase
# =============================================================================
class BayesianTriFaultBase(AdaptiveTriangularPatches, PerturbationBase):
    """
    Base class for Bayesian Triangular Faults.
    
    Inherits from:
      - AdaptiveTriangularPatches: For geometry and mesh generation.
      - PerturbationBase: For state tracking and method dispatch logic.
      
    Features:
      - Shared memory attributes for MPI efficiency.
      - Integrated help system.
    """
    def __init__(self, name: str, utmzone=None, ellps='WGS84', lon0=None, lat0=None, 
                 verbose=True, shared_info=None, use_shared_info=False, is_active=False):
        
        # 1. Bind shared properties to the class (Hijack attributes)
        self._bind_shared_properties()
        
        # 2. Setup SharedInfo object
        self.shared_info = (shared_info or SharedFaultInfo()) if use_shared_info and is_active else SharedFaultInfo()
        self.use_shared_info = use_shared_info
        self.is_active = is_active

        # 3. Initialize Parent Classes
        AdaptiveTriangularPatches.__init__(self, name, utmzone=utmzone, ellps=ellps, 
                                           lon0=lon0, lat0=lat0, verbose=verbose)
        PerturbationBase.__init__(self)

        # 4. Handle Inactive (Worker) Process State
        if use_shared_info and not is_active:
            self.shared_info = shared_info
            assert self.shared_info is not None, "shared_info is required for inactive instances."
            # Set flags to True to prevent re-computation in workers
            self.geometry_updated = True
            self.mesh_updated = True
            self.area_updated = True
            self.laplacian_updated = True

    def _bind_shared_properties(self):
        """Dynamically bind shared properties to the class."""
        for attr in SHARED_ATTRIBUTES:
            # Check if property already exists to avoid double binding in MRO
            if not isinstance(getattr(self.__class__, attr, None), property):
                setattr(self.__class__, attr, shared_property(attr))

    def copy_with_shared_info(self, name):
        """Creates a copy of the instance sharing the heavy attributes."""
        return self.__class__(name=name, shared_info=self.shared_info, use_shared_info=True, 
                              lon0=self.lon0, lat0=self.lat0, utmzone=self.utmzone, 
                              ellps=self.ellps, verbose=False, is_active=False)

    def help(self):
        """
        Prints a user-friendly list of all available perturbation methods 
        for this fault class, including methods inherited from parent classes.
        """
        info = PerturbationRegistry.get_help(self)
        
        print(f"\n{'='*20} Fault Class: {self.__class__.__name__} {'='*20}")
        print("Available Perturbation Methods:")
        
        if not info:
            print("  (No registered methods found. Please check decorators.)")
        
        for method, meta in info.items():
            print(f"  * {method}:")
            print(f"      Description: {meta.get('description', 'N/A')}")
            print(f"      Parameters:  {meta.get('params', 'N/A')}")
        print("="*60)