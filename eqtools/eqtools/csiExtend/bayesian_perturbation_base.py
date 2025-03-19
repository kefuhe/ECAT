import functools
import inspect
from .AdaptiveTriangularPatches import AdaptiveTriangularPatches
from .AdaptiveRectangularPatches import AdaptiveRectangularPatches


def track_mesh_update(update_mesh=False, update_laplacian=False, update_area=False, expected_perturbations_count=None):
    """
    Decorator function for tracking and updating the state of the mesh, Laplacian matrix, and area,
    with an option to check the count of elements in the 'perturbations' parameter.

    Args:
    update_mesh (bool): If True, the mesh update flag will be set after the function execution.
    update_laplacian (bool): If True, the Laplacian matrix update flag will be set after the function execution.
    update_area (bool): If True, the area update flag will be set after the function execution.
    expected_perturbations_count (int, optional): Expected number of elements in the 'perturbations' parameter.
                                                  If set, the function will check if the 'perturbations' parameter
                                                  matches this count.

    Returns:
    function: The decorated function.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            perturbations = args[0] if args else kwargs.get('perturbations', None)
            if perturbations is None:
                raise ValueError("The 'perturbations' parameter is required.")
            # Check the count of elements in the 'perturbations' parameter if expected_perturbations_count is set
            if expected_perturbations_count is not None and len(perturbations) != expected_perturbations_count:
                raise ValueError(f"Error: Expected {expected_perturbations_count} elements in 'perturbations', but got {len(perturbations)}.")

            self.mesh_updated = False
            self.laplacian_updated = False
            self.area_updated = False
            result = func(self, *args, **kwargs)
            if update_mesh:
                self.mesh_updated = True
            if update_laplacian:
                self.laplacian_updated = True
            if update_area:
                self.area_updated = True
            return result
        wrapper._is_decorated = True
        return wrapper
    return decorator

class PerturbationMeta(type):
    """
    Metaclass for automatically categorizing methods in a class into two categories: 
    'perturbation_methods' and 'bayesian_perturbation_methods'. 

    If a method starts with 'perturb_' and its first two parameters are 'self' and 'perturbations', 
    it is considered a Bayesian perturbation method. Otherwise, it is considered a general perturbation method. 

    This metaclass also checks if the Bayesian perturbation methods are decorated with 'track_mesh_update'. 
    If not, it raises a TypeError.
    """
    def __new__(mcs, name, bases, attrs):
        attrs['perturbation_methods'] = {}
        attrs['bayesian_perturbation_methods'] = {}
        for key, value in attrs.items():
            if callable(value) and key.startswith('perturb_'):
                parameters = list(inspect.signature(value).parameters.values())
                if len(parameters) >= 2 and parameters[0].name == 'self' and parameters[1].name == 'perturbations':
                    attrs['bayesian_perturbation_methods'][key] = attrs[key]
                    if not getattr(value, '_is_decorated', False):
                        raise TypeError(f"The method '{key}' is not decorated with 'track_mesh_update'. Please add '@track_mesh_update()' or '@track_mesh_update(True)' before the method definition.")
                else:
                    attrs['perturbation_methods'][key] = attrs[key]
        return super().__new__(mcs, name, bases, attrs)

class PerturbationBase(metaclass=PerturbationMeta):
    """
    Base class for all perturbation classes. This class uses the PerturbationMeta metaclass
    to automatically categorize perturbation methods.
    """
    def __init__(self):
        '''
        Those three flags are used to track the state of the mesh, Laplacian matrix, and area.
        If any of these flags are set to True, it means that the corresponding property has been updated and will not be recalculated in Bayesian process.
        '''
        self.geometry_updated = False
        self.mesh_updated = False
        self.laplacian_updated = False
        self.area_updated = False

    def is_mesh_updated(self):
        return self.mesh_updated
    
    def is_laplacian_updated(self):
        return self.laplacian_updated

    def perturb(self, method, **kwargs):
        """
        Perturb the geometry of the patches based on the given parameters.

        Parameters:
        method: The perturbation method to use. Must be a method of this class that starts with 'perturb_'.
        kwargs: A dictionary of arguments to pass to the perturbation method. Must include a 'perturbations' key.
        """
        if not method.startswith('perturb_'):
            raise ValueError("The method must start with 'perturb_'")

        if 'perturbations' not in kwargs:
            raise ValueError("The 'perturbations' argument is required")

        perturb_method = self.perturbation_methods.get(method, None)
        if perturb_method is None:
            available_methods = ', '.join(self.perturbation_methods.keys())
            raise ValueError(f"The method '{method}' does not exist. Available methods are: {available_methods}")

        return perturb_method(self, **kwargs)


# Define shared attributes that will be shared among all instances of BayesianTriFaultBase
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
    def getter(self):
        return getattr(self.shared_info, f"_{attr_name}")
    def setter(self, value):
        return setattr(self.shared_info, f"_{attr_name}", value)
    return property(getter, setter)

class BayesianTriFaultBase(AdaptiveTriangularPatches, PerturbationBase):
    """
    Base class for Bayesian Triangular Faults with shared attributes.

    Attributes:
        shared_info (SharedFaultInfo): Shared information among instances.
        use_shared_info (bool): Flag to indicate if shared information is used.
        is_active (bool): Flag to indicate if the fault is active.
    """
    def __init__(self, name: str, utmzone=None, ellps='WGS84', lon0=None, lat0=None, verbose=True, shared_info=None, use_shared_info=False, is_active=False):
        """
        Initialize the BayesianTriFaultBase instance.

        Args:
            name (str): Name of the fault.
            utmzone (str, optional): UTM zone.
            ellps (str, optional): Ellipsoid.
            lon0 (float, optional): Longitude.
            lat0 (float, optional): Latitude.
            verbose (bool, optional): Verbose flag.
            shared_info (SharedFaultInfo, optional): Shared information.
            use_shared_info (bool, optional): Use shared information flag.
            is_active (bool, optional): Active fault flag, default is False. It is used to indicate if the fault is active or not, and to update the shared information.
        """
        self._bind_shared_properties() # Bind shared properties to the class at first initialization
        self.shared_info = (shared_info or SharedFaultInfo()) if use_shared_info and is_active else SharedFaultInfo()
        self.use_shared_info = use_shared_info
        self.is_active = is_active

        AdaptiveTriangularPatches.__init__(self, name, utmzone=utmzone, ellps=ellps, lon0=lon0, lat0=lat0, verbose=verbose)
        PerturbationBase.__init__(self)

        if use_shared_info and not is_active:
            # Reassign the shared_info to the passed shared_info after initialization
            self.shared_info = shared_info
            assert self.shared_info is not None, "The shared_info parameter must be provided when use_shared_info is True and is_active is False."
            self.geometry_updated = True
            self.mesh_updated = True
            self.area_updated = True
            self.laplacian_updated = True

    def _bind_shared_properties(self):
        """
        Bind shared properties to the class.
        """
        for attr in SHARED_ATTRIBUTES:
            setattr(self.__class__, attr, shared_property(attr))

    def copy_with_shared_info(self, name):
        """
        Create a copy of the instance with shared information.

        Args:
            name (str): Name of the new instance.

        Returns:
            BayesianTriFaultBase: New instance with shared information.
        """
        return self.__class__(name=name, shared_info=self.shared_info, use_shared_info=True, lon0=self.lon0, lat0=self.lat0, 
                              utmzone=self.utmzone, ellps=self.ellps, verbose=False, is_active=False)