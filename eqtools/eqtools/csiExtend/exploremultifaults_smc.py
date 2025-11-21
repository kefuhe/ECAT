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
from csi import gps, insar
from .SMC_MPI import SMC_samples_parallel_mpi
from .config import explorefaultConfig
from numba import njit
from collections import namedtuple
import yaml
import glob
import os

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
                    verbose=True, fixed_params=None, config_file='default_config.yml', geodata=None, parallel_rank=None):

        self.verbose = verbose
        self.parallel_rank = parallel_rank if parallel_rank is not None else MPI.COMM_WORLD.Get_rank()
        # Initialize the fault
        if self.verbose and self.parallel_rank == 0:
            print ("---------------------------------")
            print ("---------------------------------")
            print ("Initializing fault exploration {}".format(name))

        # Base class init
        if lon0 is None or lat0 is None:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
                lon_lat_0 = config.get('lon_lat_0', None)
                if lon_lat_0:
                    lon0, lat0 = lon_lat_0
        assert lon0 is not None and lat0 is not None, "lon0 and lat0 must be set from either the config file or the arguments"

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

        # Initialize sigma values
        self._sigma_update_mask = self.config.sigmas['update'][self.config.sigmas['dataset_param_indices']] # mask for which sigmas to update
        self._sigma_initial = self.config.sigmas['values'][self.config.sigmas['dataset_param_indices']] # initial sigma values
        self._sigma_update_indices = np.where(self._sigma_update_mask)[0] # indices of sigma to be updated
        self._sigma_update_positions = self.config.sigmas['updatable_param_indices'][self.config.sigmas['dataset_param_indices']] # positions of sigma to be updated in the full parameter vector
        self._sigma_update_positions = self._sigma_update_positions[self._sigma_update_indices]
        self._sigma_update_flag = np.any(self.config.sigmas['update']) # whether any sigma is to be updated
        
        # Debug
        # print("Sigma Update Mask:", self._sigma_update_mask)
        # print("Sigma Initial Values:", self._sigma_initial)
        # print("Sigma Update Indices:", self._sigma_update_indices)
        # print("Sigma Update Positions:", self._sigma_update_positions)
        # print("Sigma Update Flag:", self._sigma_update_flag)

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
        
        if self._sigma_update_flag:
            self.param_keys['sigmas'] = []
            self.param_index['sigmas'] = []
            ndatas = self.sigmas['ndatas']
            self.sigmas_index = [len(self.Priors)+i for i in range(self.config.sigmas['updatable_params'])] # ?
            self.sigmas_keys = ['sigma_{}'.format(i) for i in range(self.config.sigmas['updatable_params'])]
            self.sigmas_keys_alias = []
            datanames = [data.name for data in self.geodata.get('data', [])]
            updatable_datanames = [datanames[i] for i in range(len(datanames)) if self._sigma_update_mask[i]]
            for i in range(self.config.sigmas['updatable_params']):
                if self.config.sigmas['mode'] == 'single':
                    self.sigmas_keys_alias.append('sigma_all')
                elif self.config.sigmas['mode'] == 'individual':
                    self.sigmas_keys_alias.append(f'sigma_{updatable_datanames[i]}')
                elif self.config.sigmas['mode'] == 'grouped':
                    group_keys = list(self.config.sigmas['groups'].keys())
                    self.sigmas_keys_alias.append(f'sigma_{group_keys[i]}')
            for i in range(self.config.sigmas['updatable_params']): # range(ndatas)
                self.param_keys['sigmas'].append(i)
                self.param_index['sigmas'].append(len(self.Priors)+i)
            bound = self.sigmas['bounds']
            for i in range(self.config.sigmas['updatable_params']): # range(ndatas)
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
                # Extract sigmas values if to be updated
                if not self._sigma_update_flag:
                    sigmas = self._sigma_initial
                else:
                    sigmas = self._sigma_initial.astype(np.float64, copy=True)
                    sigmas[self._sigma_update_indices] = samples[self.sigmas_index][self._sigma_update_positions]
                # print(f"sigmas: {sigmas}, sigmas_position: {self.sigmas_index}, _sigma_update_indices: {self._sigma_update_indices}, _sigma_update_positions: {self._sigma_update_positions}")
                # If log_scaled, convert sigmas to log scale
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
             covariance_epsilon = 1e-9, amh_a=1.0/9.0, amh_b=8.0/9.0,
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

    def returnModel(self, model='median', print_stats=True):
        '''
        Returns a list of faults corresponding to the desired model.
    
        Kwargs:
            * model             : Can be 'mean', 'median', 'std', 'MAP', an integer or a dictionary with the appropriate keys
    
        Returns:
            * list of fault instances
        '''
    
        # Get it 
        if model in ('Mean', 'mean'):
            samples = self.sampler['allsamples'].mean(axis=0)
        elif model in ('Median', 'median'):
            samples = np.median(self.sampler['allsamples'], axis=0)
        elif model in ('Std', 'std', 'STD'):                     
            samples = self.sampler['allsamples'].std(axis=0)
        elif model in ('MAP', 'map', 'Map'):
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
            if model not in ['STD', 'std', 'Std']:
                # Build the fault patches
                # Use the parameters from the samples
                fault.buildPatches(ispecs['lon'], ispecs['lat'], 
                                ispecs['depth'], ispecs['strike'],
                                ispecs['dip'], ispecs['length'],
                                ispecs['width'], 1, 1, verbose=False)
            else:
                fault.slip = np.zeros((1, 2), dtype=np.float64)  # Empty slip for STD model
            
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
        if model not in ['STD', 'std', 'Std']:
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
        if self._sigma_update_flag:
            specs['sigmas'] = np.array(samples[self.sigmas_index])

        # Save the desired model 
        if not hasattr(self, 'model_dict'):
            self.model_dict = {}
        self.model_dict[model] = specs
        self.model = samples

        # If model is not 'std', build synthetics
        if model not in ['STD', 'std', 'Std']:
            # Build Synthetics
            cogps_vertical_list = []
            cosar_list = []
            for data, vertical in zip(self.datas, self.verticals):
                if data.dtype == 'gps':
                    cogps_vertical_list.append([data, vertical])
                elif data.dtype == 'insar':
                    cosar_list.append(data)
            
            ## Build synthetics for GPS data
            for cogps, vertical in cogps_vertical_list:
                cogps.buildsynth(faults, vertical=vertical)
            
            ## Build synthetics for InSAR data
            for cosar in cosar_list:
                cosar.buildsynth(faults, vertical=True)
                # Add reference if exists
                if hasattr(cosar, 'refnumber') and '{}'.format(cosar.name) in self.keys:
                    cosar.synth += self.model[cosar.refnumber]
            
            # Calculate and print fit statistics
            if print_stats:
                self.calculate_and_print_fit_statistics(model=model)

        # All done
        return faults

    def calculate_data_fit_metrics(self, data, vertical=True):
        """
        Calculate RMS and VR for different data types.
        
        Parameters:
        -----------
        data : csi data object
            GPS, InSAR, or optical correlation data object
        vertical : bool
            Whether to include vertical component (for GPS data)
            
        Returns:
        --------
        tuple : (rms, vr)
            Root Mean Square error and Variance Reduction percentage
        """
        if data.dtype == 'insar':
            observed = data.vel
            synthetic = data.synth
        elif data.dtype == 'gps':
            if vertical:
                observed = data.vel_enu.flatten()  # Flatten all components
                synthetic = data.synth.flatten()
            else:
                observed = data.vel_enu[:, :-1].flatten()  # Only E-N components
                synthetic = data.synth[:, :-1].flatten()
        elif data.dtype in ('opticorr', 'optical'):
            observed = np.hstack((data.east, data.north))
            synthetic = np.hstack((data.east_synth, data.north_synth))
        else:
            raise ValueError(f"Unsupported data type: {data.dtype}")
        
        # Calculate RMS
        residuals = synthetic - observed
        rms = np.sqrt(np.mean(residuals**2))
        
        # Calculate Variance Reduction
        ss_res = np.sum(residuals**2)  # Sum of squares of residuals
        ss_tot = np.sum(observed**2)   # Total sum of squares
        vr = (1 - ss_res / ss_tot) * 100 if ss_tot != 0 else 0.0
        
        return rms, vr
    
    def calculate_and_print_fit_statistics(self, model='median'):
        """
        Calculate and print fit statistics for all datasets.
        
        Parameters:
        -----------
        model : str
            Model type to use ('median', 'mean', 'MAP', etc.)
        """
        # Ensure we have the model
        if model not in self.model_dict:
            faults = self.returnModel(model=model, print_stats=False)
        
        print("\n" + "="*60)
        print(f"Data Fit Statistics ({model.upper()} model)")
        print("="*60)
        
        # Build synthetics for all data
        cogps_vertical_list = []
        cosar_list = []
        for data, vertical in zip(self.datas, self.verticals):
            if data.dtype == 'gps':
                cogps_vertical_list.append([data, vertical])
            elif data.dtype == 'insar':
                cosar_list.append(data)
        
        # Calculate and print statistics
        total_rms = 0
        total_vr = 0
        data_count = 0
        
        for data, vertical in zip(self.datas, self.verticals):
            try:
                rms, vr = self.calculate_data_fit_metrics(data, vertical)
                print(f"{data.name:<15} | RMS: {rms:8.4f} | VR: {vr:6.2f}%")
                total_rms += rms
                total_vr += vr
                data_count += 1
            except Exception as e:
                print(f"{data.name:<15} | Error calculating metrics: {str(e)}")
        
        if data_count > 0:
            print("-"*60)
            print(f"{'Average':<15} | RMS: {total_rms/data_count:8.4f} | VR: {total_vr/data_count:6.2f}%")
        
        print("="*60)
    
    def save_model_to_file(self, filename=None, model='median', recalculate=False, output_to_screen=True,
                           include_std=True, include_samples=True, decimal_places=6, file_format='txt'):
        """
        Output the model parameters to a file and/or screen in a beautiful format.
    
        Args:
            * filename  : The name of the file to write the model parameters to (default is None)
    
        Kwargs:
            * model     : 'mean', 'median', 'std', 'MAP'
            * recalculate: True/False
            * output_to_screen: True/False, whether to output to screen (default is True)
            * include_std: True/False, whether to include std values (default is True)
            * include_samples: True/False, whether to include raw samples array (default is True)
            * decimal_places: int, number of decimal places for values (default is 6)
            * file_format: str, output file format ('json', 'yaml', 'csv', 'txt') (default is 'json')
    
        Returns:
            * None
        """
        from tabulate import tabulate
        import json
        import yaml
        import csv
    
        # Get estimated parameters info
        estimated_params = self.print_mcmc_parameter_positions(print_table=False)
    
        # Ensure we have the required models
        if model != 'std' and (not hasattr(self, 'model_dict') or 'std' not in self.model_dict):
            self.returnModel(model='std', print_stats=False)
    
        if recalculate or model not in self.model_dict:
            self.returnModel(model=model, print_stats=False)
    
        # Get the samples for the specified model
        if model in ('mean', 'Mean'):
            samples = self.sampler['allsamples'].mean(axis=0)
        elif model in ('median', 'Median'):
            samples = np.median(self.sampler['allsamples'], axis=0)
        elif model in ('std', 'Std', 'STD'):
            samples = self.sampler['allsamples'].std(axis=0)
        elif model in ('MAP', 'map', 'Map'):
            max_posterior_index = np.argmax(self.sampler['postval'])
            samples = self.sampler['allsamples'][max_posterior_index, :]
        else:
            raise ValueError(f"Unsupported model type: {model}")
    
        # Prepare data for tabular display (for screen output)
        table_data = []
        index_counter = 0
    
        # Prepare structured data for file output
        structured_data = {
            'metadata': {
                'model_type': model.upper(),
                'total_estimated_parameters': 0,
                'include_std': include_std,
                'decimal_places': decimal_places,
                'generation_info': {
                    'fixed_parameters_note': "Parameters marked with '*' are fixed values (not estimated)",
                    'fixed_parameters_index': "Fixed parameters have Index='N/A' and STD=0.000000"
                }
            },
            'parameters': {
                'faults': {},
                'reference': {},
                'sigmas': {}
            }
        }
    
        if include_samples:
            structured_data['raw_samples'] = {
                'description': f'Raw samples array ({model.upper()})',
                'values': [round(float(sample), decimal_places) for sample in samples]
            }
    
        # Add fault parameters
        for fault_name in self.faultnames:
            fault_params = self.model_dict[model].get(fault_name, {})
            std_params = self.model_dict.get('std', {}).get(fault_name, {}) if include_std else {}
            fixed_params = self.fixed_params.get(fault_name, {})
            
            # Get estimated parameters for this fault
            estimated_fault_params = estimated_params['fault'].get(fault_name, [])
            
            structured_data['parameters']['faults'][fault_name] = {
                'estimated': {},
                'fixed': {}
            }
            
            # Iterate through parameters in the fixed order (self.keys)
            for param in self.keys:
                if param in estimated_fault_params:
                    # Estimated parameter (appears in MCMC)
                    value = fault_params[param]
                    std_value = std_params.get(param, None) if include_std else None
                    
                    param_data = {
                        'index': index_counter,
                        'value': round(float(value), decimal_places) if isinstance(value, (int, float)) else value
                    }
                    
                    if include_std and std_value is not None:
                        param_data['std'] = round(float(std_value), decimal_places) if isinstance(std_value, (int, float)) else std_value
                    
                    structured_data['parameters']['faults'][fault_name]['estimated'][param] = param_data
                    
                    # For screen display
                    row = [
                        index_counter,
                        'Fault',
                        fault_name,
                        param,
                        f"{value:.{decimal_places}f}" if isinstance(value, (int, float)) else str(value)
                    ]
                    
                    if include_std and std_value is not None:
                        row.append(f"{std_value:.{decimal_places}f}" if isinstance(std_value, (int, float)) else 'N/A')
                    elif include_std:
                        row.append('N/A')
                    
                    table_data.append(row)
                    index_counter += 1
                    
                elif param in fixed_params:
                    # Fixed parameter
                    value = fixed_params[param]
                    
                    structured_data['parameters']['faults'][fault_name]['fixed'][param] = {
                        'value': round(float(value), decimal_places) if isinstance(value, (int, float)) else value,
                        'std': 0.0 if include_std else None
                    }
                    
                    # For screen display
                    param_name = f"{param}*"
                    row = [
                        'N/A',
                        'Fault',
                        fault_name,
                        param_name,
                        f"{value:.{decimal_places}f}" if isinstance(value, (int, float)) else str(value)
                    ]
                    
                    if include_std:
                        row.append(f"{0:.{decimal_places}f}")
                    
                    table_data.append(row)
    
        # Add reference parameters
        if 'reference' in self.model_dict[model]:
            ref_values = self.model_dict[model]['reference']
            std_ref_values = self.model_dict.get('std', {}).get('reference', []) if include_std else []
            
            for i, (ref_name, ref_value) in enumerate(zip(self.param_keys['reference'], ref_values)):
                std_value = std_ref_values[i] if i < len(std_ref_values) else None
                
                param_data = {
                    'index': index_counter,
                    'value': round(float(ref_value), decimal_places) if isinstance(ref_value, (int, float)) else ref_value
                }
                
                if include_std and std_value is not None:
                    param_data['std'] = round(float(std_value), decimal_places) if isinstance(std_value, (int, float)) else std_value
                
                structured_data['parameters']['reference'][ref_name] = param_data
                
                # For screen display
                row = [
                    index_counter,
                    'Reference',
                    ref_name,
                    'reference',
                    f"{ref_value:.{decimal_places}f}" if isinstance(ref_value, (int, float)) else str(ref_value)
                ]
                
                if include_std:
                    row.append(f"{std_value:.{decimal_places}f}" if isinstance(std_value, (int, float)) and std_value is not None else 'N/A')
                
                table_data.append(row)
                index_counter += 1
    
        # Add sigma parameters
        if 'sigmas' in self.model_dict[model]:
            sigma_values = self.model_dict[model]['sigmas']
            std_sigma_values = self.model_dict.get('std', {}).get('sigmas', []) if include_std else []
            
            for i, (update_idx, sigma_value) in enumerate(zip(self.param_keys['sigmas'], sigma_values)):
                if self.config.sigmas['mode'] == 'individual':
                    isigma_name = self.datas[self._sigma_update_indices[update_idx]].name
                elif self.config.sigmas['mode'] == 'single':
                    isigma_name = 'All data'
                elif self.config.sigmas['mode'] == 'grouped':
                    i_index = np.where(self.config.sigmas['update'])[0][update_idx]
                    ikey = list(self.config.sigmas["groups"].keys())[i_index]
                    isigma_name = f'{ikey}' + ' (' + ', '.join(self.config.sigmas["groups"][ikey]) + ')'
                std_value = std_sigma_values[i] if i < len(std_sigma_values) else None
                
                param_data = {
                    'index': index_counter,
                    'data_name': isigma_name,
                    'value': round(float(sigma_value), decimal_places) if isinstance(sigma_value, (int, float)) else sigma_value
                }
                
                if include_std and std_value is not None:
                    param_data['std'] = round(float(std_value), decimal_places) if isinstance(std_value, (int, float)) else std_value
                
                structured_data['parameters']['sigmas'][f'sigma_{update_idx}'] = param_data
                
                # For screen display
                row = [
                    index_counter,
                    'Sigma',
                    isigma_name,
                    'sigma',
                    f"{sigma_value:.{decimal_places}f}" if isinstance(sigma_value, (int, float)) else str(sigma_value)
                ]
                
                if include_std:
                    row.append(f"{std_value:.{decimal_places}f}" if isinstance(std_value, (int, float)) and std_value is not None else 'N/A')
                
                table_data.append(row)
                index_counter += 1
    
        # Update metadata
        structured_data['metadata']['total_estimated_parameters'] = index_counter
        structured_data['metadata']['total_parameters'] = len(table_data)
    
        # Screen output (unchanged)
        if output_to_screen:
            headers = ['Index', 'Category', 'Name', 'Parameter', model.upper()]
            if include_std and model.lower() != 'std':
                headers.append('STD')
    
            table_str = tabulate(table_data, headers=headers, tablefmt='grid', stralign='left', floatfmt=f'.{decimal_places}f')
            
            print("=" * 80)
            print(f"MCMC Model Parameters Summary ({model.upper()})")
            print("=" * 80)
            print("")
            print("Note: Parameters marked with '*' are fixed values (not estimated)")
            print("      Fixed parameters have Index='N/A' and STD=0.000000")
            print("")
            print(table_str)
            print(f"\nTotal estimated parameters: {index_counter}")
            
            if include_samples:
                print(f"\nRaw samples array ({model.upper()}):")
                samples_rounded = [round(float(sample), decimal_places) for sample in samples]
                print(f"{samples_rounded}")
            
            print("=" * 80)
    
        # File output with different formats
        if filename:
            # Determine file format from extension if not specified
            if file_format == 'auto':
                file_ext = filename.split('.')[-1].lower()
                if file_ext in ['json']:
                    file_format = 'json'
                elif file_ext in ['yml', 'yaml']:
                    file_format = 'yaml'
                elif file_ext in ['csv']:
                    file_format = 'csv'
                else:
                    file_format = 'txt'
    
            if file_format == 'json':
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(structured_data, f, indent=2, ensure_ascii=False)
            
            elif file_format == 'yaml':
                with open(filename, 'w', encoding='utf-8') as f:
                    yaml.dump(structured_data, f, default_flow_style=False, allow_unicode=True, indent=2)
            
            elif file_format == 'csv':
                with open(filename, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    headers = ['Index', 'Category', 'Name', 'Parameter', model.upper()]
                    if include_std and model.lower() != 'std':
                        headers.append('STD')
                    writer.writerow(headers)
                    writer.writerows(table_data)
            
            elif file_format == 'txt':
                # Simple text format for easy reading
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(f"MCMC Model Parameters Summary ({model.upper()})\n")
                    f.write("=" * 60 + "\n\n")
                    
                    # Write fault parameters
                    for fault_name in self.faultnames:
                        f.write(f"Fault: {fault_name}\n")
                        f.write("-" * 30 + "\n")
                        
                        fault_data = structured_data['parameters']['faults'][fault_name]
                        
                        if fault_data['estimated']:
                            f.write("Estimated parameters:\n")
                            for param, data in fault_data['estimated'].items():
                                if include_std and 'std' in data:
                                    f.write(f"  {param:<12}: {data['value']:<12.{decimal_places}f} ± {data['std']:.{decimal_places}f}\n")
                                else:
                                    f.write(f"  {param:<12}: {data['value']:.{decimal_places}f}\n")
                        
                        if fault_data['fixed']:
                            f.write("Fixed parameters:\n")
                            for param, data in fault_data['fixed'].items():
                                f.write(f"  {param}*<11>: {data['value']:.{decimal_places}f}\n")
                        
                        f.write("\n")
                    
                    # Write reference parameters
                    if structured_data['parameters']['reference']:
                        f.write("Reference parameters:\n")
                        f.write("-" * 30 + "\n")
                        for ref_name, data in structured_data['parameters']['reference'].items():
                            if include_std and 'std' in data:
                                f.write(f"  {ref_name:<12}: {data['value']:<12.{decimal_places}f} ± {data['std']:.{decimal_places}f}\n")
                            else:
                                f.write(f"  {ref_name:<12}: {data['value']:.{decimal_places}f}\n")
                        f.write("\n")
                    
                    # Write sigma parameters
                    if structured_data['parameters']['sigmas']:
                        f.write("Sigma parameters:\n")
                        f.write("-" * 30 + "\n")
                        for sigma_name, data in structured_data['parameters']['sigmas'].items():
                            if include_std and 'std' in data:
                                f.write(f"  {data['data_name']:<12}: {data['value']:<12.{decimal_places}f} ± {data['std']:.{decimal_places}f}\n")
                            else:
                                f.write(f"  {data['data_name']:<12}: {data['value']:.{decimal_places}f}\n")
                        f.write("\n")
                    
                    # Write raw samples if requested
                    if include_samples:
                        f.write("Raw samples array:\n")
                        f.write("-" * 30 + "\n")
                        samples_str = str(structured_data['raw_samples']['values'])
                        f.write(f"{samples_str}\n")
    
            if output_to_screen:
                print(f"\nModel parameters saved to: {filename} (format: {file_format})")
    
        return estimated_params
    
    def plot(self, model='median', show=True, scale=2., legendscale=0.5, vertical=True):
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
        faults = self.returnModel(model=model, print_stats=False)

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

    def plot_kde_matrix(self, figsize=None, save=False, filename='kde_matrix.png', show=True, 
                        style='white', fill=True, scatter=False, scatter_size=15, 
                        plot_sigmas=False, plot_faults=True, faults=None, axis_labels=None,
                        wspace=None, hspace=None, center_lon_lat=False,
                        xtick_rotation=None, ytick_rotation=None, lonlat_decimal=3,
                        use_sigma_alias=True,
                        # Font size control - split into tick and label
                        tick_fontsize=None, label_fontsize=None,
                        # Tick marks control
                        show_minor_ticks=False, tick_direction='in',
                        major_tick_length=3, minor_tick_length=1.5,
                        tick_width=0.5):
        """
        Plot a KDE matrix of the SMC samples.
    
        Parameters:
        figsize: tuple, optional
            Figure size in inches. The default is (7.5, 6.5).
        save: bool, optional
            Whether to save the figure. The default is False.
        filename: str, optional
            The name of the file to save the figure. The default is 'kde_matrix.png'.
        show: bool, optional
            Whether to show the figure. The default is True.
        style: str, optional
            The style of the plot. The default is 'white'.
        fill: bool, optional
            Whether to fill the KDE plots. The default is True.
        scatter: bool, optional
            Whether to plot scatter points on the upper half. The default is False.
        scatter_size: int, optional
            The size of the scatter points. The default is 15.
        plot_sigmas: bool, optional
            Whether to plot sigmas. The default is False.
        plot_faults: bool, optional
            Whether to plot faults. The default is True.
        faults: list or str, optional
            The list of faults to plot. The default is None.
        axis_labels: list, optional
            The list of axis labels. The default is None.
        wspace: float, optional
            The amount of width reserved for space between subplots, expressed as a fraction of the average axis width. The default is None.
        hspace: float, optional
            The amount of height reserved for space between subplots, expressed as a fraction of the average axis height. The default is None.
        center_lon_lat: bool, optional
            Whether to center lon and lat by subtracting their means. The default is False.
        xtick_rotation: int, optional
            The rotation of x-tick labels. The default is None.
        ytick_rotation: int, optional
            The rotation of y-tick labels. The default is None.
        lonlat_decimal: int, optional
            The number of decimal places to round lon and lat. The default is 3.
        use_sigma_alias: bool, optional
            Whether to use sigma aliases. The default is True.
        tick_fontsize: float, optional
            Font size for tick labels. The default is None.
        label_fontsize: float, optional
            Font size for axis labels. The default is None.
        show_minor_ticks: bool, optional
            Whether to show minor tick marks. The default is False.
        tick_direction: str, optional
            Direction of tick marks ('in', 'out', 'inout'). The default is 'in'.
        major_tick_length: float, optional
            Length of major tick marks in points. The default is 3.
        minor_tick_length: float, optional
            Length of minor tick marks in points. The default is 1.5.
        tick_width: float, optional
            Width of tick marks in points. The default is 0.5.
    
        Returns:
        None
        """
        import seaborn as sns
        import pandas as pd
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.ticker import FuncFormatter, AutoLocator
    
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
        
        if plot_sigmas and hasattr(self, 'sigmas_keys') and hasattr(self, 'sigmas_index'):
            # Convert sigma keys to LaTeX format for better display
            if use_sigma_alias:
                sigma_keys = []
                for key in self.sigmas_keys_alias:
                    if key.startswith('sigma_'):
                        # Convert sigma_xxx to $\sigma_{xxx}$
                        subscript = key.replace('sigma_', '')
                        latex_key = f'$\\sigma_{{{subscript}}}$'
                        sigma_keys.append(latex_key)
                    else:
                        sigma_keys.append(key)
            else:
                sigma_keys = []
                for key in self.sigmas_keys:
                    if key.startswith('sigma_'):
                        # Convert sigma_xxx to $\sigma_{xxx}$
                        subscript = key.replace('sigma_', '')
                        latex_key = f'$\\sigma_{{{subscript}}}$'
                        sigma_keys.append(latex_key)
                    else:
                        sigma_keys.append(key)
            
            keys += sigma_keys
            index += self.sigmas_index

        # Replace 'magnitude' with 'slip' in all keys (unified processing)
        keys = [key.replace('magnitude', 'slip') for key in keys]
        # Capitalize all keys for better display
        keys = [key.capitalize() for key in keys]
        # Convert the SMC chains to a DataFrame
        df = pd.DataFrame(trace[:, index], columns=keys)
        
        # Remove columns with zero variance
        df = df.loc[:, df.var() != 0]
        
        # Center lon and lat if required
        if center_lon_lat:
            for key in keys:
                if 'lon' in key:
                    lon_mean = df[key].mean()
                    df[key] -= lon_mean
                    print(f"Mean of {key}: {lon_mean}")
                if 'lat' in key:
                    lat_mean = df[key].mean()
                    df[key] -= lat_mean
                    print(f"Mean of {key}: {lat_mean}")
        
        # Set the style
        sns.set_style(style)
        
        # Set PDF font type if saving as PDF
        if save and filename.endswith('.pdf'):
            pdf_fonttype = 42  # Use Type 42 (TrueType) for better compatibility
            plt.rcParams['pdf.fonttype'] = pdf_fonttype
        
        # Create a pair grid with separate y-axis for diagonal plots
        g = sns.PairGrid(df, diag_sharey=False)
        
        if figsize is not None:
            g.figure.set_size_inches(*figsize)
        
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
        
        # Configure tick marks for all subplots
        for i in range(len(g.axes)):
            for j in range(len(g.axes)):
                if g.axes[i, j].get_visible():
                    # Enable or disable minor ticks
                    if show_minor_ticks:
                        g.axes[i, j].minorticks_on()
                    else:
                        g.axes[i, j].minorticks_off()
                    
                    # Configure major tick marks
                    g.axes[i, j].tick_params(
                        axis='both',
                        which='major',
                        direction=tick_direction,
                        length=major_tick_length,
                        width=tick_width,
                        top=False,
                        right=False,
                        bottom=True,
                        left=True
                    )
                    
                    # Configure minor tick marks (only if enabled)
                    if show_minor_ticks:
                        g.axes[i, j].tick_params(
                            axis='both',
                            which='minor',
                            direction=tick_direction,
                            length=minor_tick_length,
                            width=tick_width,
                            top=False,
                            right=False,
                            bottom=True,
                            left=True
                        )
                    
                    # Ensure tick locators are set
                    g.axes[i, j].xaxis.set_major_locator(AutoLocator())
                    g.axes[i, j].yaxis.set_major_locator(AutoLocator())
        
        # Format tick labels to scientific notation for lon and lat
        def scientific_formatter(x, pos):
            return f'{x:.2g}' if x <= 1 else f'{x:.{lonlat_decimal}f}'
        
        for ax in g.axes.flatten():
            if ax is not None and ax.get_visible():
                if 'lon' in ax.get_xlabel() or 'lat' in ax.get_xlabel():
                    ax.xaxis.set_major_formatter(FuncFormatter(scientific_formatter))
                    for label in ax.get_xticklabels():
                        label.set_rotation(45)
                        label.set_ha('right')
                if 'lon' in ax.get_ylabel() or 'lat' in ax.get_ylabel():
                    ax.yaxis.set_major_formatter(FuncFormatter(scientific_formatter))
    
        # Set tick rotation and font size
        default_tick_fontsize = tick_fontsize if tick_fontsize is not None else 10
        if xtick_rotation is not None:
            for ax in g.axes[-1, :]:
                for label in ax.get_xticklabels():
                    label.set_rotation(xtick_rotation)
                    label.set_ha('right')
                    label.set_fontsize(default_tick_fontsize)
        else:
            for ax in g.axes[-1, :]:
                ax.tick_params(axis='x', labelsize=default_tick_fontsize)
    
        if ytick_rotation is not None:
            for ax in g.axes[:, 0]:
                for label in ax.get_yticklabels():
                    label.set_rotation(ytick_rotation)
                    label.set_ha('right')
                    label.set_fontsize(default_tick_fontsize)
        else:
            for ax in g.axes[:, 0]:
                ax.tick_params(axis='y', labelsize=default_tick_fontsize)
        
        # Set axis labels if provided
        default_label_fontsize = label_fontsize if label_fontsize is not None else 12
        if axis_labels:
            for i, label in enumerate(axis_labels):
                g.axes[-1, i].set_xlabel(label, fontsize=default_label_fontsize)
                g.axes[i, 0].set_ylabel(label, fontsize=default_label_fontsize)
        else:
            # Set fontsize for existing axis labels
            for i in range(len(g.axes)):
                if g.axes[-1, i].get_xlabel():
                    g.axes[-1, i].set_xlabel(g.axes[-1, i].get_xlabel(), fontsize=default_label_fontsize)
                if g.axes[i, 0].get_ylabel():
                    g.axes[i, 0].set_ylabel(g.axes[i, 0].get_ylabel(), fontsize=default_label_fontsize)
    
        plt.tight_layout()
        if wspace is not None or hspace is not None:
            plt.subplots_adjust(wspace=wspace, hspace=hspace)
        
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
                                        antisymmetric=True, res_use_data_norm=True, cmap='jet',
                                        model='median', fault_figsize=(7.5, 6.5), sigmas_figsize=(7.5, 6.5),
                                        save_data=True, sar_corner=None):
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
        model: the model to use ('mean', 'median', 'std', 'MAP', default is 'mean')
        fault_figsize: figure size for fault KDE plots (default is A4 size (7.5, 6.5))
        sigmas_figsize: figure size for sigmas KDE plots (default is A4 size (7.5, 6.5))
        save_data: whether to save the data to a file (default is True)
        sar_corner (None, 'tri', 'quad'): sar corner type (default is None)
        """
        if model == 'std':
            plot_data = False

        if rank == 0:
            self.load_samples_from_h5(filename=filename)
            self.print_mcmc_parameter_positions(print_table=False)
            
            # Plot Faults
            if plot_faults:
                for ifault, faultname in enumerate(self.faultnames):
                    self.plot_kde_matrix(save=True, plot_faults=True, faults=faultname, fill=True, 
                                        scatter=False, filename=f'kde_matrix_F{ifault}.png', figsize=fault_figsize,
                                        hspace=0.05, wspace=0.05)
            
            # Plot Sigmas
            if plot_sigmas and hasattr(self, 'sigmas_keys') and hasattr(self, 'sigmas_index'):
                self.plot_kde_matrix(save=True, plot_faults=False, plot_sigmas=True, fill=True, 
                                    scatter=False, filename='kde_matrix_sigmas.png', figsize=sigmas_figsize,
                                    hspace=0.05, wspace=0.05)
            
            # Save the model results
            faults = self.returnModel(model=model, print_stats=False)
            self.save_model_to_file(f'model_results_{model}.txt', model=model, output_to_screen=True)
            self.calculate_and_print_fit_statistics(model=model)

            # Build synthetics for GPS and SAR data
            cogps_vertical_list = []
            cosar_list = []
            for data, vertical in zip(self.datas, self.verticals):
                if data.dtype == 'gps':
                    cogps_vertical_list.append([data, vertical])
                elif data.dtype == 'insar':
                    cosar_list.append(data)

            if save_data:
                if not os.path.exists('Modeling'):
                    os.makedirs('Modeling')
                # Save SAR data
                if sar_corner is not None:
                    for i, sardata in enumerate(cosar_list):
                        corner_flag = True if sar_corner=='tri' else False
                        for itype in ['data', 'synth', 'resid']:
                            sardata.writeDecim2file(f'{sardata.name}_{itype}.txt', itype, outDir='Modeling', triangular=corner_flag)
                else:
                    for i, sardata in enumerate(cosar_list):
                        for itype in ['data', 'synth', 'resid']:
                            sardata.write2file(f'{sardata.name}_{itype}.txt', itype, outDir='Modeling')
                # Save GPS data
                for i, (gpsdata, gpsvertical) in enumerate(cogps_vertical_list):
                    for itype in ['data', 'synth', 'res']:
                        gpsdata.write2file(f'{gpsdata.name}_{itype}.txt', itype, outDir='Modeling')
            
            # Plot GPS and SAR data
            if plot_data:
                # Plot GPS data
                for fault in faults:
                    fault.color = 'b' # Set the color to blue
                for cogps, vertical in cogps_vertical_list:
                    # cogps.buildsynth(faults, vertical=vertical)
                    box = [cogps.lon.min(), cogps.lon.max(), cogps.lat.min(), cogps.lat.max()]
                    cogps.plot(faults=faults, drawCoastlines=True, data=['data', 'synth'], scale=0.2, legendscale=0.05, color=['k', 'r'],
                            seacolor='lightblue', box=box, titleyoffset=1.02)
                    cogps.fig.savefig(f'gps_{cogps.name}', ftype='png', dpi=600, 
                                    bbox_inches='tight', mapaxis=None, saveFig=['map'])
                
                # Plot SAR data
                for fault in faults:
                    fault.color = 'k'
                for cosar in cosar_list:
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
    
    def print_mcmc_parameter_positions(self, print_table=True):
        """Print the MCMC parameter positions and return estimated parameters info."""
        from tabulate import tabulate
        
        # Collect all parameter information
        all_params = []
        estimated_params = {
            'fault': {},  # {fault_name: [param_names]}
            'reference': [],
            'sigmas': []
        }
        
        # Add fault parameters
        for fault_name in self.faultnames:
            estimated_params['fault'][fault_name] = []
            for i, key in enumerate(self.param_keys[fault_name]):
                all_params.append([
                    'Fault',
                    fault_name,
                    key,
                    self.param_index[fault_name][i]
                ])
                estimated_params['fault'][fault_name].append(key)
        
        # Add reference parameters
        if 'reference' in self.param_keys:
            for i, key in enumerate(self.param_keys['reference']):
                all_params.append([
                    'Reference',
                    key,
                    'reference',
                    self.param_index['reference'][i]
                ])
                estimated_params['reference'].append(key)
        
        # Add sigma parameters
        if 'sigmas' in self.param_keys:
            for i, ikey in enumerate(self.param_keys['sigmas']):
                idata = self._sigma_update_indices[ikey]
                iname = f'data_{idata}' if self.datas is None else self.datas[idata].name
                all_params.append([
                    'Sigma',
                    iname,
                    'sigma',
                    self.param_index['sigmas'][ikey]
                ])
                estimated_params['sigmas'].append(iname)
        
        # Sort by index
        all_params.sort(key=lambda x: x[3])
        
        # Print beautiful table if requested
        if print_table:
            headers = ['Category', 'Name', 'Parameter', 'Index']
            print("\nMCMC Parameter Positions:")
            print(tabulate(all_params, headers=headers, tablefmt='grid', stralign='left'))
            print(f"\nTotal parameters: {len(all_params)}")
        
        return estimated_params

    def guess_initial_dimensions_and_slip(self, magnitude, fault_type='SS'):
        """
        Guess initial fault length, width, and slip magnitude based on magnitude and fault type.

        Parameters:
        magnitude (float): The magnitude of the earthquake.
        fault_type (str): The type of the fault ('SS' for strike-slip, 'R' for reverse-slip, 'N' for normal-slip).

        Returns:
        dict: A dictionary containing the guessed length, width, and slip magnitude.
        """
        from ..Tectonic_Utils.seismo import wells_and_coppersmith as wac

        length = wac.RLD_from_M(magnitude, fault_type)
        width = wac.RW_from_M(magnitude, fault_type)
        # slip = wac.SLR_from_M(magnitude, fault_type)
        slip_magnitude = wac.rectangular_slip(length * 1e3, width * 1e3, magnitude)  # Convert length and width to meters

        return {
            'length': length,
            'width': width,
            'slip': slip_magnitude,
        }

#EOF
