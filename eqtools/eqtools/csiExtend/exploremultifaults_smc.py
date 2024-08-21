'''
A class that searches for the best fault to fit some geodetic data.
This class is made for a simple planar fault geometry.
It is close to what R. Grandin has implemented but with a MCMC approach
Grandin's approach will be coded in another class.

Author:
R. Jolivet 2017

Modifications:
Changed by Kefeng He on 2023-11-16 for the purpose of exploring multiple faults

and change pymc to SMP-MPI for parallel sampling, which is more efficient for large number of parameters
'''

# Externals
from mpi4py import MPI
import json
try:
    import h5py
except:
    print('No hdf5 capabilities detected')
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform

# Personals
from csi import SourceInv
from csi import planarfault
from .SMC_MPI import SMC_samples_parallel_mpi
from numba import njit
from collections import namedtuple
import yaml

@njit
def logpdf_multivariate_normal(x, mean, inv_cov, logdet):
    norm_const = -0.5 * logdet
    x_mu = np.subtract(x, mean)
    solution = np.dot(inv_cov, x_mu)
    result = -0.5 * np.dot(x_mu, solution) + norm_const
    return result

@njit
def compute_data_log_likelihood(simulations, observations, inv_cov, log_cov_det):
    data_log_likelihood = logpdf_multivariate_normal(observations, simulations, inv_cov, log_cov_det)
    return data_log_likelihood

@njit
def compute_log_prior(samples, lb, ub):
    if np.any((samples < lb) | (samples > ub)):
        return -np.inf
    else:
        return 0.0

NT1 = namedtuple('NT1', 'N Neff target LB UB')
# tuple object for the samples
NT2 = namedtuple('NT2', 'allsamples postval beta stage covsmpl resmpl')


class explorefaultConfig:
    def __init__(self, config_file=None, geodata=None):
        self.nchains = 100 # Number of chains for BayesianMultiFaultsInversion
        self.chain_length = 50 # Length of each chain for BayesianMultiFaultsInversion
        self.geodata = {} # Dictionary of geodetic data
        self.bounds = {}
        self.initial = {}
        self.geodata = {}
        self.fixed_params = {}
        self.nfaults = 1
        self.faultnames = [f'fault_{i}' for i in range(self.nfaults)]
        self.dataFaults = None
        self.slip_sampling_mode = 'mag_rake'

        if config_file:
            self.load_config(config_file, geodata=geodata)

    def load_config(self, config_file, geodata=None):
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        self.bounds = config.get('bounds', {})
        self.initial = config.get('initial', {})
        self.fixed_params = config.get('fixed_params', {})
        self.nfaults = config.get('nfaults', 1)
        self.faultnames = [f'fault_{i}' for i in range(self.nfaults)]
        self.slip_sampling_mode = config.get('slip_sampling_mode', 'mag_rake')
        self.clipping_options = config.get('clipping_options', {})
        self.geodata = config.get('geodata', {})

        self._update_geodata(geodata)
        self._validate_verticals()
        self._set_geodata_attributes()
        self.update_polys_estimate_and_boundaries()
        self._select_data_sets()

    def _update_geodata(self, geodata):
        if 'data' not in self.geodata or self.geodata['data'] is None:
            self.geodata['data'] = geodata if geodata else []
        assert self.geodata['data'], 'No geodata provided, need to provide at least one data set'

    def _validate_verticals(self):
        verticals = self.geodata.get('verticals', None)
        if verticals is None:
            verticals = True
        data_length = len(self.geodata['data'])
        if isinstance(verticals, list):
            if len(verticals) != data_length:
                raise ValueError(f"Length of 'verticals' list ({len(verticals)}) does not match length of 'data' ({data_length})")
        elif isinstance(verticals, bool):
            self.geodata['verticals'] = [verticals] * data_length
        else:
            raise ValueError("'verticals' must be either a list or a boolean")

    def _set_geodata_attributes(self):
        '''
        To trigger the property setters for sigmas and dataFaults
        '''
        self.sigmas = self.geodata.get('sigmas', {})
        self.dataFaults = self.geodata.get('faults', None)

    def _select_data_sets(self):
        data_verticals_dict = {d.name: v for d, v in zip(self.geodata['data'], self.geodata['verticals'])}
        if self.clipping_options.get('enabled', False):
            if self.clipping_options.get('method', 'lon_lat_range') == 'lon_lat_range':
                lon_lat_range = self.clipping_options.get('lon_lat_range', None)
                if lon_lat_range is None:
                    raise ValueError("Clipping method 'lon_lat_range' requires 'lon_lat_range' to be set")
                for data in self.geodata['data']:
                    if data.dtype == 'insar':
                        data.select_pixels(*lon_lat_range)
                    elif data.dtype == 'gps':
                        data.select_stations(*lon_lat_range)
                        if not data_verticals_dict[data.name]:
                            data.vel_enu[:, -1] = np.nan
                            data.buildCd(direction='en')
                        else:
                            data.buildCd(direction='enu')

    def update_polys_estimate_and_boundaries(self, datas=None):
        if self.geodata.get('polys', {}).get('enabled', False):
            if datas is not None:
                if type(datas) is not list:
                    datas = [datas]
                self.geodata['polys']['estimate'] = [d.name for d in datas]
            else:
                datas = self.geodata.get('polys', {}).get('estimate', [])

            insar_data = [d for d in self.geodata.get('data', []) if d.dtype == 'insar']
            default_bounds = self.geodata['polys']['boundaries'].get('defaults', None)
            
            if not datas:
                datas = [d.name for d in insar_data]
                self.geodata['polys']['estimate'] = datas
            
            for data in insar_data:
                if data.name in datas:
                    boundary_key = data.name
                    if boundary_key not in self.geodata['polys']['boundaries']:
                        if default_bounds:
                            self.geodata['polys']['boundaries'][boundary_key] = default_bounds
                        else:
                            raise ValueError(f"Bounds for {boundary_key} must be set as there is no default")
                else:
                    raise ValueError(f"Data name {data.name} is not in the estimate list")

    @property
    def dataFaults(self):
        return self.geodata['faults']
    
    @dataFaults.setter
    def dataFaults(self, value):
        if value is None:
            self.geodata['faults'] = [self.faultnames] * len(self.geodata.get('data', []))
        elif isinstance(value, list):
            # Flatten the list of lists to a single list if sublist is a list
            flattened_dataFaults = [item for sublist in value for item in (sublist if isinstance(sublist, list) else [sublist])]
            
            # Check flattened_dataFaults is subset of self.faultnames
            if not set(flattened_dataFaults).issubset(set(self.faultnames + [None])):
                raise ValueError("The dataFaults must be a subset of the faultnames in self.multifaults")
            
            # Ensure the list is at most two levels deep
            for sublist in value:
                if isinstance(sublist, list):
                    if any(isinstance(item, list) for item in sublist):
                        raise ValueError("The dataFaults list must be at most two levels deep")
                    if all(item is None for item in sublist) or all(item is not None for item in sublist):
                        continue
                    else:
                        raise ValueError("The second level lists must either contain only None or no None")
            
            # Replace None with faultnames in geodata['faults']
            self.geodata['faults'] = [[item if item is not None else self.faultnames for item in sublist] if isinstance(sublist, list) else (sublist if sublist is not None else self.faultnames) for sublist in value]
        else:
            raise ValueError("dataFaults must be a list or None")

    @property
    def sigmas(self):
        return self.geodata['sigmas']
    
    @sigmas.setter
    def sigmas(self, value):
        self.geodata['sigmas'] = value
        self.geodata['sigmas']['ndatas'] = self.ndatas
        self.geodata['sigmas']['names'] = ['sigma_{}'.format(i) for i in range(self.ndatas)]

        sigma_names = self.geodata['sigmas']['names']
        bounds = self.geodata['sigmas']['bounds']
    
        # Check if 'defaults' is in bounds
        if 'defaults' not in bounds:
            # Check if all sigmas are in bounds
            if not set(sigma_names).issubset(bounds.keys()):
                raise ValueError("The bounds dictionary must have keys for all sigmas or a 'defaults' key")
        else:
            # Fill in the missing sigmas with the defaults
            defaults = bounds['defaults']
            for name in sigma_names:
                if name not in bounds:
                    bounds[name] = defaults
    
    @property
    def ndatas(self):
        return len(self.geodata.get('data', []))


# Class explorefault
class explorefault(SourceInv):
    '''
    Creates an object that will solve for the best fault details. The fault has only one patch and is embedded in an elastic medium.

    Args:
        * name          : Name of the object

    Kwargs:
        * utmzone       : UTM zone number
        * ellps         : Ellipsoid
        * lon0/lat0     : Refernece of the zone
        * verbose       : Talk to me
        * fixed_params  : A nested dictionary of parameters to be fixed. The outer dictionary's keys are the fault names, and the values are another dictionary where the keys are the parameter names and the values are the fixed values.

    Returns:
        * None
    '''

    def __init__(self, name, mode=None, num_faults=None, utmzone=None, 
                    ellps='WGS84', lon0=None, lat0=None, 
                    verbose=True, fixed_params=None, config_file='default_config.yml', geodata=None):

        # Initialize the fault
        if verbose:
            print ("---------------------------------")
            print ("---------------------------------")
            print ("Initializing fault exploration {}".format(name))
        self.verbose = verbose

        # Base class init
        super(explorefault, self).__init__(name, utmzone=utmzone, 
                                            ellps=ellps, 
                                            lon0=lon0, lat0=lat0)

        # Load and set the configuration
        self.load_and_set_config(config_file, fixed_params, geodata=geodata)

        # Initialize the fault objects
        self.nfaults = num_faults if num_faults else self.nfaults
        self.faults = {f'fault_{i}': planarfault('mcmc fault {}'.format(i), utmzone=self.utmzone, 
                                                lon0=self.lon0, 
                                                lat0=self.lat0,
                                                ellps=self.ellps, 
                                                verbose=False) for i in range(self.nfaults)}
        self.faultnames = [f'fault_{i}' for i in range(len(self.faults))]

        # Keys to look for
        self.mode = mode if mode else self.slip_sampling_mode
        if self.mode == 'ss_ds':
            self.keys = ['lon', 'lat', 'depth', 'dip', 
                            'width', 'length', 'strike', 
                            'strikeslip', 'dipslip']
        elif self.mode == 'mag_rake':
            self.keys = ['lon', 'lat', 'depth', 'dip', 
                            'width', 'length', 'strike', 
                            'magnitude', 'rake']
        else:
            raise ValueError("Invalid mode. Expected 'ss_ds' or 'mag_rake'.")

        # Initialize the index for each fault's parameters
        self.param_index = {}
        self.param_keys = {}
        self.total_params = 0
        index = 0
        for i in range(self.nfaults):
            fault_name = f'fault_{i}'
            self.param_index[fault_name] = []
            self.param_keys[fault_name] = []
            for param in self.keys:
                if param not in self.fixed_params.get(fault_name, {}):
                    self.param_index[fault_name].append(index)
                    self.param_keys[fault_name].append(param)
                    self.total_params += 1
                    index += 1

        # All done
        return
    
    def load_and_set_config(self, config_file, fixed_params, geodata=None):
        # Load the configuration file
        self.config = explorefaultConfig(config_file, geodata=geodata)
        
        # Set fixed parameters
        self.fixed_params = self.config.fixed_params if self.config.fixed_params else {}
        if fixed_params:
            self.fixed_params.update(fixed_params)
        
        # Set other configuration parameters
        self.nchains = self.config.nchains
        self.chain_length = self.config.chain_length
        self.bounds = self.config.bounds
        self.geodata = self.config.geodata
        self.sigmas = self.config.sigmas
        self.dataFaults = self.config.dataFaults
        self.ndatas = self.config.ndatas
        self.nfaults = self.config.nfaults
        self.slip_sampling_mode = self.config.slip_sampling_mode
    
    def build_fault_params(self, samples, fault_name):
        '''
        Build the parameters for a fault.
    
        Args:
            * samples      : A numpy array of samples
            * fault_name   : The name of the fault
    
        Returns:
            * params       : A dictionary of parameter values
        '''
        indices = self.param_index[fault_name]
        keys = self.param_keys[fault_name]
        random_params = dict(zip(keys, samples[indices]))
        fixed_params = self.fixed_params.get(fault_name, {})
    
        # Filter the keys
        random_params = {k: v for k, v in random_params.items() if k in self.keys}
        fixed_params = {k: v for k, v in fixed_params.items() if k in self.keys}
    
        return {**random_params, **fixed_params}
    
    def setPriors(self, bounds=None, datas=None, initialSample=None, sigmas=None):
        '''
        Initializes the prior likelihood functions for multiple faults.

        Args:
            * bounds        : Bounds is a dictionary that holds the following keys. 
                   - 'defaults' : A dictionary for default fault parameters that holds the following keys.
                       - 'lon'        : Longitude (tuple or float)
                       - 'lat'        : Latitude (tuple or float)
                       - 'depth'      : Depth in km of the top of the fault (tuple or float)
                       - 'dip'        : Dip in degree (tuple or float)
                       - 'width'      : Along-dip size in km (tuple or float)
                       - 'length'     : Along-strike length in km (tuple or float)
                       - 'strike'     : Azimuth of the strike (tuple or float)
                       - ss_ds mode:
                           - 'strikeslip' : Strike Slip (tuple or float)
                           - 'dipslip'    : Dip slip (tuple or float)
                       - mag_rake mode:
                           - 'magnitude'  : Magnitude (tuple or float)
                           - 'rake'       : Rake (tuple or float)
                   - 'fault_name' : A dictionary for each fault that holds the same keys as 'defaults'. These will override the defaults.
                   - 'data_name'  : A dictionary for each data set that holds the following keys.
                       - 'reference' : Reference value (tuple or float)

                              One bound should be a list with the name of a pymc distribution as first element. The following elements will be passed on to the function.
                              example:  bounds[0] = ('Normal', 0., 2.) will give a Normal distribution centered on 0. with a 2. standard deviation.

        Kwargs:
            * datas         : Data sets that will be used. This is in case bounds has tuples or floats for reference of an InSAR data set

            * initialSample : An array the size of the list of bounds default is None and will be randomly set from the prior PDFs

        Returns:
            * None
        '''
        if bounds is None:
            if (not hasattr(self, 'bounds') or len(self.bounds)==0):
                raise ValueError("No bounds provided and no bounds attribute found.")
        else:
            if not hasattr(self, 'bounds') or len(self.bounds)==0:
                self.bounds = bounds
            else:
                self.bounds.update(bounds)
        
        bounds = self.bounds

        # Make a list of priors
        if not hasattr(self, 'Priors'):
            self.Priors = []

        # Check initialSample
        if initialSample is None:
            initialSample = {}
        else:
            assert len(initialSample)==len(bounds), \
                'Inconsistent size for initialSample: {}'.format(len(initialSample))
        initSampleVec = []

        # Iterate over the faults
        for fault_name in self.faultnames:

            # Merge the default bounds with the fault-specific bounds
            fault_bounds = bounds.get(fault_name, {})
            default_bounds = bounds.get('defaults', {})
            merged_bounds = {**default_bounds, **fault_bounds}

            # Iterate over the keys
            for key in self.param_keys[fault_name]:

                # Get the values
                bound = merged_bounds[key]

                # Get the function type
                assert type(bound[0]) is str, 'First element of a bound must be a string'
                if bound[0] == 'Normal':
                    # Use scipy's normal distribution instead of numpy's
                    function = norm
                elif bound[0] == 'Uniform':
                    # Use scipy's uniform distribution instead of numpy's
                    function = uniform
                else:
                    raise ValueError(f"Invalid distribution type: {bound[0]}. Only 'Uniform' and 'Normal' are allowed.")

                # Get arguments and create the prior
                args = bound[1:]
                pm_func = function(*args)

                # Initial Sample
                ikey = f"{fault_name}_{key}"
                initialSample.setdefault(ikey, pm_func.rvs())  # draw a sample for the initial sample

                # Save it
                if bound[0]!='Degenerate':
                    self.Priors.append(pm_func)
                    initSampleVec.append(initialSample[ikey])

        # Create a prior for the data set reference term
        # Works only for InSAR data yet
        self.config.update_polys_estimate_and_boundaries(datas)
        if self.config.geodata.get('polys', {}).get('enabled', False):
            datas = self.config.geodata['polys']['estimate']
            self.dataReferences = datas
            datas = [d for d in self.config.geodata.get('data', []) if d.name in datas]
            self.param_keys['reference'] = []
            self.param_index['reference'] = []
                
            # Iterate over the data
            for data in datas:
                
                # Get it
                assert data.name in self.config.geodata['polys']['boundaries'], \
                    'No bounds provided for prior for data {}'.format(data.name)
                bound = self.config.geodata['polys']['boundaries'][data.name]
                key = '{}'.format(data.name)

                # Get the function type
                assert type(bound[0]) is str, 'First element of a bound must be a string'
                if bound[0] == 'Normal':
                    # Use scipy's normal distribution instead of numpy's
                    function = norm
                elif bound[0] == 'Uniform':
                    # Use scipy's uniform distribution instead of numpy's
                    function = uniform
                else:
                    raise ValueError(f"Invalid distribution type: {bound[0]}. Only 'Uniform' and 'Normal' are allowed.")

                # Get arguments and create the prior
                args = bound[1:]
                pm_func = function(*args)

                # Initial Sample
                initialSample.setdefault(key, pm_func.rvs())  # draw a sample for the initial sample

                # Store it
                if bound[0]!='Degenerate':
                    self.Priors.append(pm_func)
                    initSampleVec.append(initialSample[key])
                    self.keys.append(key)
                    self.param_keys['reference'].append(key)
                    self.param_index['reference'].append(len(self.Priors)-1)
                data.refnumber = len(self.Priors)-1
        
        # Set Sigmas priors
        if sigmas is not None:
            if not hasattr(self, 'Sigmas') or len(self.Sigmas)==0:
                self.sigmas = sigmas
            else:
                self.sigmas.update(sigmas)
        else:
            if not hasattr(self, 'sigmas') or len(self.sigmas)==0:
                self.sigmas = {}
        
        if self.sigmas['update']:
            self.param_keys['sigmas'] = []
            self.param_index['sigmas'] = []
            ndatas = self.sigmas['ndatas']
            self.sigmas_index = [len(self.Priors)+i for i in range(ndatas)]
            self.sigmas_keys = ['sigma_{}'.format(i) for i in range(ndatas)]
            for i in range(ndatas):
                self.param_keys['sigmas'].append(i)
                self.param_index['sigmas'].append(len(self.Priors)+i)
            bound = self.sigmas['bounds']
            for i in range(ndatas):
                ibound = bound['defaults'] if self.sigmas_keys[i] not in bound else bound[self.sigmas_keys[i]]
                if ibound[0] == 'Normal':
                    # Use scipy's normal distribution instead of numpy's
                    function = norm
                elif ibound[0] == 'Uniform':
                    # Use scipy's uniform distribution instead of numpy's
                    function = uniform
                else:
                    raise ValueError(f"Invalid distribution type: {ibound[0]}. Only 'Uniform' and 'Normal' are allowed.")
                args = ibound[1:]
                pm_func = function(*args)
                self.Priors.append(pm_func)
                ikey = f'sigma_{i}'
                initialSample.setdefault(ikey, pm_func.rvs())  # draw a sample for the initial sample
                initSampleVec.append(initialSample[ikey])
                # initSampleVec.append(pm_func.rvs())
        else:
            self.sigmas_values = self.sigmas['values']

        # Save initial sample
        self.initSampleVec = initSampleVec
        self.initialSample = initialSample

        # All done
        return

    def setLikelihood(self, datas=None, verticals=None):
        '''
        Builds the data likelihood object from the list of geodetic data in datas.
    
        Args:   
            * datas         : csi geodetic data object (gps or insar) or list of csi geodetic objects. TODO: Add other types of data (opticorr)
    
        Kwargs:
            * verticals      : A list of booleans indicating whether to use the vertical component of the data.
    
        Returns:
            * None
        '''
    
        # Build the prediction method
        # Initialize the object
        self.datas = datas if datas else self.geodata['data']
        self.verticals = verticals if verticals else self.geodata.get('verticals', [True]*len(self.datas))
    
        # List of likelihoods
        self.Likelihoods = []
    
        # Create a likelihood function for each of the data set
        for data, vertical in zip(self.datas, self.verticals):
    
            # Get the data type
            if data.dtype=='gps':
                # Get data
                if vertical:
                    dobs = data.vel_enu.flatten()
                else:
                    dobs = data.vel_enu[:,:-1].flatten()
            elif data.dtype=='insar':
                # Get data
                dobs = data.vel
    
            # Make sure Cd exists
            assert hasattr(data, 'Cd'), \
                    'No data covariance for data set {}'.format(data.name)
            Cd = data.Cd
            Cd_inv = np.linalg.inv(Cd)
            # Cd_det = np.linalg.det(Cd)
            # logCd_det = np.log(Cd_det)
            # logCd_det = np.log(np.linalg.det(Cd))
            sign, logdet = np.linalg.slogdet(Cd)
            logCd_det = sign * logdet
            ilike = [dobs, Cd_inv, logCd_det]
            # Save the likelihood function
            self.Likelihoods.append([data]+ [il.astype(np.float64) for il in ilike] + [vertical])
    
        # All done 
        return

    def Predict(self, theta, data, vertical=True, faultnames=None, updatepatch=True):
        if hasattr(data, 'refnumber'):
            reference = theta[data.refnumber]
        else:
            reference = 0.
    
        # Get the fault
        if faultnames is None:
            faultnames = self.faultnames
        # else:
        #     faultnames = list(set(self.faultnames).intersection(faultnames))
        faults = [self.faults[fault_name] for fault_name in faultnames]
        for fault_name in faultnames:
            fault = self.faults[fault_name]
            # Take the values in theta and distribute
            params = self.build_fault_params(theta, fault_name)
            lon, lat, depth, strike, dip, length, width = params['lon'], params['lat'], params['depth'], params['strike'], params['dip'], params['length'], params['width']
    
            # Build a planar fault
            if updatepatch:
                fault.buildPatches(lon, lat, depth, strike, dip, length, width, 1, 1, verbose=False)
    
            # Build the green's functions
            fault.buildGFs(data, vertical=vertical, slipdir='sd', verbose=False)
    
            if self.mode == 'ss_ds':
                strikeslip, dipslip = params['strikeslip'], params['dipslip']
            elif self.mode == 'mag_rake':
                magnitude, rake = params['magnitude'], params['rake']
                rad_rake = np.radians(rake)
                strikeslip = magnitude*np.cos(rad_rake)
                dipslip = magnitude*np.sin(rad_rake)
    
            # Set slip 
            fault.slip[:,0] = strikeslip
            fault.slip[:,1] = dipslip

        # Build empty Green's functions for the data where faults is not in faultnames but in self.faultnames
        for fault_name in set(self.faultnames).difference(faultnames):
            fault = self.faults[fault_name]
            fault.buildGFs(data, vertical=vertical, slipdir='sd', verbose=False, method='empty')
    
        # Build the synthetics
        data.buildsynth(faults)
    
        # check data type 
        if data.dtype=='gps':
            if vertical: 
                return data.synth.flatten()
            else:
                return data.synth[:,:-1].flatten()
        elif data.dtype=='insar':
            return data.synth.flatten()+reference
    
        # All done
        return
    
    def make_target(self, updatepatches=None, dataFaults=None):
        # Extract lb and ub from self.Priors
        self.lb = np.array([prior.args[0] for prior in self.Priors])
        self.ub = np.array([prior.args[0] + prior.args[1] for prior in self.Priors])
        
        if dataFaults is None:
            dataFaults = [None]*len(self.Likelihoods)
        
        # Initialize updatepatches based on dataFaults
        if updatepatches is None:
            updatepatches = [False]*len(self.Likelihoods)  # Default is False
        
            faults_union = set()
            for i, faults in enumerate(dataFaults):
                if faults is None:
                    updatepatches[i] = True
                    break  # Found a complete set, stop further checks
                else:
                    faults_union.update(faults)
                    if faults_union >= set(self.faultnames):
                        updatepatches[i] = True
                        break  # Found a complete set through combination, stop further checks
    
        def target(samples):
            log_prior = compute_log_prior(samples, self.lb, self.ub)
            if log_prior == -np.inf:
                return -np.inf
            else:
                log_likelihood = 0
                sigmas = self.sigmas_values if not self.sigmas['update'] else samples[self.sigmas_index]
                sigmas = np.power(10, np.array(sigmas)) if self.sigmas['log_scaled'] else np.array(sigmas)
                for i, (data, dobs, Cd_inv, logCd_det, vertical) in enumerate(self.Likelihoods):
                    # Determine the faults to use for this dataset
                    currentFaults = self.faultnames if dataFaults[i] is None else dataFaults[i]
                    simulations = self.Predict(samples, data, vertical=vertical, 
                                               faultnames=currentFaults, updatepatch=updatepatches[i])
                    isigma2 = sigmas[i]**2
                    Cd_inv_sigma = np.divide(Cd_inv, isigma2)
                    logCd_det_sigma = logCd_det + np.log(isigma2)*len(dobs)
                    log_likelihood += compute_data_log_likelihood(simulations, dobs, Cd_inv_sigma, logCd_det_sigma)
                return log_prior + log_likelihood
        return target
    
    def walk(self, nchains=None, chain_length=None, comm=None, filename='samples.h5',
             save_every=1, save_at_interval=True, save_at_final=True,
             covariance_epsilon = 1e-6, amh_a=1.0/9.0, amh_b=8.0/9.0,
             updatepatches=None, dataFaults=None):
        '''
        March the SMC.
    
        Kwargs:
            * nchains           : Number of Markov Chains
            * chain_length      : Length of each chain
            * print_samples     : Whether to print the samples
            * filename          : The name of the HDF5 file to save the samples
            * dataFaults        : A list of faults to use for each data set. 
                                    If None, all faults will be used for all data sets.
            * updatepatches     : A list of booleans indicating whether to update the patches for each data set. 
                                    If None, the patches will be updated for all data sets.
    
        Returns:
            * None
        '''
        nchains = nchains if nchains else self.nchains
        chain_length = chain_length if chain_length else self.chain_length

        self.dataFaults = dataFaults or self.dataFaults
        # Create a target function
        target = self.make_target(updatepatches=updatepatches, dataFaults=self.dataFaults)
    
        # Create an NT1 object
        opt = NT1(nchains, chain_length, target, self.lb, self.ub)
    
        # Create an NT2 object for the samples
        samples = NT2(None, None, None, None, None, None)
    
        # Get the MPI rank
        if comm is None:
            comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
    
        if rank == 0:
            print('Starting the loop...', flush=True)
    
        # Run the SMC sampling
        final = SMC_samples_parallel_mpi(opt, samples, NT1, NT2, comm, save_at_final, 
                                         save_every, save_at_interval, covariance_epsilon, amh_a, amh_b)
    
        if rank == 0:
            # Save the final samples
            self.sampler = final._asdict()
            # Save the samples to an HDF5 file
            self.save2h5(filename)
            print('Finished the loop.')
    
        # All done
        return

    def returnModels(self, model='mean'):
        '''
        Returns a list of faults corresponding to the desired model.
    
        Kwargs:
            * model             : Can be 'mean', 'median', 'std', 'MAP', an integer or a dictionary with the appropriate keys
    
        Returns:
            * list of fault instances
        '''
    
        # Get it 
        if model=='mean':
            samples = self.sampler['allsamples'].mean(axis=0)
        elif model=='median':
            samples = self.sampler['allsamples'].median(axis=0)
        elif model=='std':                     
            samples = self.sampler['allsamples'].std(axis=0)
        elif model=='MAP':
            # Assuming 'logposterior' is the key for log posterior values
            max_posterior_index = np.argmax(self.sampler['postval'])
            samples = self.sampler['allsamples'][max_posterior_index, :]
        else: 
            if type(model) is int:
                samples = self.sampler['allsamples'][model,:]
            else:
                assert type(model) is int, 'Model type unknown: {}'.format(model)

        # Create a list to store the faults
        faults = []
        specs = {}

        # Iterate over the fault names
        for fault_name in self.faultnames:

            ispecs = self.build_fault_params(samples, fault_name)

            specs[fault_name] = ispecs

            fault = self.faults[fault_name]
            fault.buildPatches(ispecs['lon'], ispecs['lat'], 
                               ispecs['depth'], ispecs['strike'],
                               ispecs['dip'], ispecs['length'],
                               ispecs['width'], 1, 1, verbose=False)
            
            # Set slip values
            if self.mode == 'mag_rake':
                fault.slip[:,0] = ispecs['magnitude']*np.cos(np.radians(ispecs['rake']))
                fault.slip[:,1] = ispecs['magnitude']*np.sin(np.radians(ispecs['rake']))
            elif self.mode == 'ss_ds':
                fault.slip[:,0] = ispecs['strikeslip']
                fault.slip[:,1] = ispecs['dipslip']

            # Add the fault to the list
            faults.append(fault)
        
        # Build Green's functions for the data based on self.dataFaults
        for idf, idataFaults in enumerate(self.dataFaults):
            if idataFaults is None:
                self.dataFaults[idf] = self.faultnames
        for fault_name in self.faultnames:
            fault = self.faults[fault_name]
            for ilike in self.Likelihoods:
                # Build the green's functions
                fault.buildGFs(ilike[0], slipdir='sd', verbose=False, vertical=ilike[-1])
        # Build different set between self.faultnames and self.dataFaults
        for idataFaults, ilike in zip(self.dataFaults, self.Likelihoods):
            for fault_name in set(self.faultnames).difference(idataFaults):
                fault = self.faults[fault_name]
                fault.buildGFs(ilike[0], slipdir='sd', verbose=False, method='empty', vertical=ilike[-1])
        
        # Extract the reference values
        if 'reference' in self.param_keys:
            specs['reference'] = np.array(samples[self.param_index['reference']])

        # Extract the sigmas
        if self.sigmas['update']:
            specs['sigmas'] = np.array(samples[self.sigmas_index])

        # Save the desired model 
        if not hasattr(self, 'model_dict'):
            self.model_dict = {}
        self.model_dict[model] = specs
        self.model = samples

        # All done
        return faults
    
    def save_model_to_file(self, filename, model='mean', recalculate=False):
        """
        Output the model parameters to a file.

        Args:
            * filename  : The name of the file to write the model parameters to

        Kwargs:
            * model     : 'mean', 'median', 'std', 'MAP'
            * recalculate: True/False
        
        Returns:
            * None
        """
        if recalculate or model not in self.model_dict:
            self.returnModels(model=model)

        with open(filename, 'w', encoding='utf-8') as file:
            # 写入 Fault 参数
            for fault_name, fault_params in self.model_dict[model].items():
                if fault_name.startswith('fault_'):
                    file.write(f"Fault: {fault_name}\n")
                    for param, value in fault_params.items():
                        file.write(f"  {param}: {value}\n")
                    file.write("\n")
            
            # 写入 Reference 参数
            if 'reference' in self.model_dict[model]:
                file.write("Reference:\n")
                for ref_name, ref_value in zip(self.param_keys['reference'], self.model_dict[model]['reference']):
                    file.write(f"  {ref_name}: {ref_value}\n")
                file.write("\n")
            
            # 写入 Sigmas 参数
            if 'sigmas' in self.model_dict[model]:
                file.write("Sigmas:\n")
                for data, sigma_name, sigma_value in zip(self.datas, self.param_keys['sigmas'], self.model_dict[model]['sigmas']):
                    isigma_name = data.name
                    file.write(f"  {isigma_name}: {sigma_value}\n")
                file.write("\n")
    
    def plot(self, model='mean', show=True, scale=2., legendscale=0.5, vertical=True):
        '''
        Plots the PDFs and the desired model predictions and residuals.

        Kwargs:
            * model     : 'mean', 'median' or 'rand'
            * show      : True/False

        Returns:
            * None
        '''

        # Plot the pymc stuff
        for iprior, prior in enumerate(self.Priors):
            trace = self.sampler['allsamples'][:,iprior]
            fig = plt.figure()
            plt.subplot2grid((1,4), (0,0), colspan=3)
            plt.plot([0, len(trace)], [trace.mean(), trace.mean()], 
                     '--', linewidth=2)
            plt.plot(trace, 'o-')
            plt.title(self.keys[iprior])
            plt.subplot2grid((1,4), (0,3), colspan=1)
            plt.hist(trace, orientation='horizontal')
            #plt.savefig('{}.png'.format(prior[0]))

        # Get the model
        faults = self.returnModels(model=model)

        # Build predictions
        for fault in self.faults:
            for data, vertical in zip(self.datas, self.verticals):

                # Build the green's functions
                fault.buildGFs(data, slipdir='sd', verbose=False, vertical=vertical)

        # Build the synthetics
        data.buildsynth(fault)

            # Check ref
        for data in self.datas:
            if '{}'.format(data.name) in self.keys:
                data.synth += self.model[data.refnumber] # ['{}'.format(data.name)]

            # Plot the data and synthetics
            if data.dtype == 'insar':
                cmin = np.min(data.vel)
                cmax = np.max(data.vel)
                data.plot(data='data', norm=[cmin, cmax], show=False)
                data.plot(data='synth', norm=[cmin, cmax], show=False)
            elif data.dtype == 'gps':
                data.plot(data=['synth', 'data'], color=['r', 'k'], scale=scale, drawCoastlines=False, legendscale=legendscale)
        
        # Plot
        if show:
            plt.show()

        # All done
        return

    def plot_smc_statistics(self):
        import arviz as az
        import matplotlib.pyplot as plt

        # Get the SMC chains
        trace = self.sampler['allsamples']
        keys = self.keys

        # Convert the SMC chains to a dictionary
        data = {keys[i]: trace[:, i] for i in range(len(keys))}

        # Convert the dictionary to an InferenceData object
        idata = az.from_dict(data)

        # Plot the sample traces
        az.plot_trace(idata)

        # Plot the posterior distributions
        az.plot_posterior(idata)

        # Plot the autocorrelation
        az.plot_autocorr(idata)

        plt.show()

    def plot_kde_matrix(self, figsize=(7.5, 6.5), save=False, filename='kde_matrix.png', show=True, 
                        style='white', fill=True, scatter=False, scatter_size=15, 
                        plot_sigmas=False, plot_faults=True, faults=None):
        import seaborn as sns
        import pandas as pd
        import matplotlib.pyplot as plt
        import numpy as np
    
        # Get the SMC chains
        trace = self.sampler['allsamples']
        keys = []
        index = []
        if plot_faults:
            if faults is None:
                for fault_name in self.faultnames:
                    keys += [f"{fault_name}_{key}" for key in self.param_keys[fault_name]]
                    index += self.param_index[fault_name]
            elif type(faults) in (list, ):
                for fault_name in faults:
                    keys += [f"{fault_name}_{key}" for key in self.param_keys[fault_name]]
                    index += self.param_index[fault_name]
            elif type(faults) in (str, ):
                assert faults in self.faultnames, f"Fault {faults} not found."
                keys += self.param_keys[faults]
                index += self.param_index[faults]
        
        if plot_sigmas:
            keys += self.sigmas_keys
            index += self.sigmas_index
        
        # Convert the SMC chains to a DataFrame
        df = pd.DataFrame(trace[:, index], columns=keys)
    
        # Remove columns with zero variance
        df = df.loc[:, df.var() != 0]

        # Set the style
        sns.set_style(style)

        # Create a pair grid with separate y-axis for diagonal plots
        g = sns.PairGrid(df, diag_sharey=False)

        # Remove the upper half of plots if scatter is not required
        if not scatter:
            for i, j in zip(*np.triu_indices_from(g.axes, 1)):
                g.axes[i, j].set_visible(False)

        # Plot a filled KDE on the diagonal
        g.map_diag(sns.kdeplot, fill=fill)

        # Plot a filled KDE with scatter points on the off-diagonal
        g.map_lower(sns.kdeplot, fill=fill)

        # Plot scatter points on the upper half if required
        if scatter:
            g.map_upper(sns.scatterplot, s=scatter_size)

        plt.tight_layout()
        # Save the figure if required
        if save:
            plt.savefig(filename, dpi=600)

        # Show the figure if required
        if show:
            plt.show()

    def plot_smc_statistics_arviz(self):
        import arviz as az
        import matplotlib.pyplot as plt
        # plt.rcParams['plot.max_subplots'] = 100
    
        # Get the SMC chains
        trace = self.sampler['allsamples']
        keys = self.keys
    
        # Convert the SMC chains to a dictionary
        data = {keys[i]: trace[:, i] for i in range(len(keys))}
    
        # Convert the dictionary to an InferenceData object
        idata = az.from_dict(data)
    
        # Plot the pair plot
        az.plot_pair(idata, marginals=True)
    
        plt.show()

    def extract_and_plot_bayesian_results(self, rank=0, filename='samples_mag_rake_multifaults.h5', 
                                        plot_faults=True, plot_sigmas=True, plot_data=True,
                                        antisymmetric=True, res_use_data_norm=True, cmap='jet'):
        """
        Extract and plot the Bayesian results.

        args:
        rank: process rank (default is 0)
        filename: name of the HDF5 file to save the samples (default is 'samples_mag_rake_multifaults.h5')
        plot_faults: whether to plot faults (default is True)
        plot_sigmas: whether to plot sigmas (default is True)
        plot_data: whether to plot data (default is True)
        antisymmetric: whether to set the colormap to be antisymmetric (default is True)
        res_use_data_norm: whether to make the norm of 'res' consistent with 'data' and 'synth' (default is True)
        cmap: colormap to use (default is 'jet')
        """
        if rank == 0:
            self.load_samples_from_h5(filename=filename)
            self.print_mcmc_parameter_positions()
            
            # Plot Faults
            if plot_faults:
                for ifault, faultname in enumerate(self.faultnames):
                    self.plot_kde_matrix(save=True, plot_faults=True, faults=faultname, fill=True, 
                                        scatter=False, filename=f'kde_matrix_F{ifault}.png')
            
            # Plot Sigmas
            if plot_sigmas:
                self.plot_kde_matrix(save=True, plot_faults=False, plot_sigmas=True, fill=True, 
                                    scatter=False, filename='kde_matrix_sigmas.png')
            
            # save the model results
            faults = self.returnModels(model='mean')
            self.save_model_to_file('model_results_mean.json', model='mean')

            if plot_data:
                cogps_vertical_list = []
                cosar_list = []
                for data, vertical in zip(self.datas, self.verticals):
                    if data.dtype == 'gps':
                        cogps_vertical_list.append([data, vertical])
                    elif data.dtype == 'insar':
                        cosar_list.append(data)
                
                # Plot GPS data
                for fault in faults:
                    fault.color = 'b' # Set the color to blue
                for cogps, vertical in cogps_vertical_list:
                    cogps.buildsynth(faults, vertical=vertical)
                    box = [cogps.lon.min(), cogps.lon.max(), cogps.lat.min(), cogps.lat.max()]
                    cogps.plot(faults=faults, drawCoastlines=True, data=['data', 'synth'], scale=0.2, legendscale=0.05, color=['k', 'r'],
                            seacolor='lightblue', box=box, titleyoffset=1.02)
                    cogps.fig.savefig(f'gps_{cogps.name}', ftype='png', dpi=600, 
                                    bbox_inches='tight', mapaxis=None, saveFig=['map'])
                
                # Plot SAR data
                for fault in faults:
                    fault.color = 'k'
                for cosar in cosar_list:
                    cosar.buildsynth(faults, vertical=True)
                    datamin, datamax = cosar.vel.min(), cosar.vel.max()
                    absmax = max(abs(datamin), abs(datamax))
                    data_norm = [-absmax, absmax] if antisymmetric else [datamin, datamax]
                    for data in ['data', 'synth', 'res']:
                        if data == 'res':
                            cosar.res = cosar.vel - cosar.synth
                            absmax = max(abs(cosar.res.min()), abs(cosar.res.max()))
                            res_norm = [-absmax, absmax] if antisymmetric else [cosar.res.min(), cosar.res.max()]
                            res_norm = data_norm if res_use_data_norm else res_norm
                            cosar.plot(faults=faults, data=data, seacolor='lightblue', figsize=(3.5, 2.7), norm=res_norm, cmap=cmap,
                                cbaxis=[0.15, 0.25, 0.25, 0.02], drawCoastlines=True, titleyoffset=1.02)
                        else:
                            cosar.plot(faults=faults, data=data, seacolor='lightblue', figsize=(3.5, 2.7), norm=data_norm, cmap=cmap,
                                    cbaxis=[0.15, 0.25, 0.25, 0.02], drawCoastlines=True, titleyoffset=1.02)
                        cosar.fig.savefig(f'sar_{cosar.name}_{data}', ftype='png', dpi=600, saveFig=['map'], 
                                        bbox_inches='tight', mapaxis=None)
            
            print(self.model_dict)

    def save2h5(self, filename, datasets=None):
        '''
        Save samples to an HDF5 file.
    
        Args:
            * filename  : The name of the HDF5 file
            * datasets  : A list of dataset names to save. If None, all datasets will be saved.
    
        Returns:
            * None
        '''
    
        # If no datasets are specified, save all datasets
        if datasets is None:
            datasets = ['allsamples', 'postval', 'beta', 'stage', 'covsmpl', 'resmpl']
    
        # Get the samples
        samples = self.sampler
    
        try:
            # Open the HDF5 file
            with h5py.File(filename, 'w') as f:
                # Save each dataset
                for dataset in datasets:
                    data = samples[dataset]
                    f.create_dataset(dataset, data=data)
        except Exception as e:
            print(f'Error saving to HDF5 file: {e}')
    
    def load_samples_from_h5(self, filename, datasets=None):
        '''
        Load samples from an HDF5 file.

        Args:
            * filename  : The name of the HDF5 file
            * datasets  : A list of dataset names to load. If None, all datasets will be loaded.

        Returns:
            * None
        '''

        # If no datasets are specified, load all datasets
        if datasets is None:
            datasets = ['allsamples', 'postval', 'beta', 'stage', 'covsmpl', 'resmpl']

        # Create a dictionary to hold the samples
        samples = {}

        try:
            # Open the HDF5 file
            with h5py.File(filename, 'r') as f:
                # Load each dataset
                for dataset in datasets:
                    data = f[dataset][:]
                    samples[dataset] = data
        except Exception as e:
            print(f'Error loading from HDF5 file: {e}')

        # Save the loaded samples
        self.sampler = samples
    
    def print_mcmc_parameter_positions(self):
        """Print the MCMC parameter positions."""
        print("MCMC parameter positions:")
        for ifault in self.faultnames:
            print(f"Fault: {ifault}")
            for ikey, key in enumerate(self.param_keys[ifault]):
                print(f"  {key}: {self.param_index[ifault][ikey]}")
        if 'reference' in self.param_keys:
            print('Reference:')
            for ikey, key in enumerate(self.param_keys['reference']):
                print(f"  {key}: {self.param_index['reference'][ikey]}")
        if 'sigmas' in self.param_keys:
            print('Sigmas:')
            for ikey, key in enumerate(self.param_keys['sigmas']):
                iname = 'data_{}'.format(ikey) if self.datas is None else self.datas[ikey].name
                print(f"  {iname}: {self.param_index['sigmas'][ikey]}")

#EOF
