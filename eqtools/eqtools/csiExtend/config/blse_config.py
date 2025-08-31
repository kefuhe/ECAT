from .linear_config import LinearInversionConfig
from ..multifaults_base import MyMultiFaultsInversion


class BoundLSEInversionConfig(LinearInversionConfig):
    """
    Bounded Least Squares Estimation (BLSE) inversion configuration.
    
    This class provides configuration management specifically for bounded least squares 
    inversion, inheriting common linear inversion functionality from LinearInversionConfig.
    """
    
    def __init__(self, config_file='default_config.yml', multifaults=None, geodata=None, 
                 verticals=None, polys=None, dataFaults=None, alphaFaults=None, faults_list=None,
                 gfmethods=None, encoding='utf-8', verbose=False, **kwargs):
        """
        Initialize the BoundLSEInversionConfig object.
        
        Parameters:
        -----------
        config_file : str, optional
            Path to the configuration file (default: 'default_config.yml')
        multifaults : object, optional
            Multifaults object for the inversion
        geodata : list, optional
            List of geodetic data objects
        verticals : list, optional
            List of vertical displacement flags for each dataset
        polys : list, optional
            List of polynomial correction orders for each dataset
        dataFaults : list, optional
            List of fault names for each dataset
        alphaFaults : list, optional
            List of alpha parameter groups for faults
        faults_list : list, optional
            List of fault objects
        gfmethods : list, optional
            List of Green's function methods for each fault
        encoding : str, optional
            File encoding for configuration file (default: 'utf-8')
        verbose : bool, optional
            Enable verbose output (default: False)
        **kwargs : dict
            Additional keyword arguments
        """
        self._sigmas_param_name = 'initial_value'  # BLSE uses 'initial_value' for sigmas
        # Call parent class initialization
        super().__init__(config_file=config_file, multifaults=multifaults, geodata=geodata,
                        verticals=verticals, polys=polys, dataFaults=dataFaults, 
                        alphaFaults=alphaFaults, faults_list=faults_list, gfmethods=gfmethods,
                        encoding=encoding, verbose=verbose, **kwargs)
        
        # BLSE-specific configuration
        # self.alpha['update'] = False  # BLSE does not update alpha by default
        
        # BLSE-specific validation
        self._validate_laplacian_bounds()
        
        # Data assembly
        if self.clipping_options.get('enabled', False):
            self._initialize_faults_and_assemble_data()

        self._initialize_faults_and_assemble_data()
        
        # Initialize multifaults object
        if multifaults is None:
            multifaults = MyMultiFaultsInversion('myfault', self.faults_list, verbose=False)
            self.multifaults = multifaults
        multifaults.assembleGFs()

        if self.parallel_rank is None or self.parallel_rank == 0:
            self.export_config()

    def set_attributes(self, **kwargs):
        """
        Set object attributes based on key-value pairs in kwargs.
        
        Parameters:
        -----------
        **kwargs : dict
            Dictionary of attribute names and values to set
        """
        # Set the attributes based on the key-value pairs in kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)