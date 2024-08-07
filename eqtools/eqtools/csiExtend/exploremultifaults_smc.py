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

    def __init__(self, name, mode='ss_ds', num_faults=1, utmzone=None, 
                    ellps='WGS84', lon0=None, lat0=None, 
                    verbose=True, fixed_params=None, config_file=None):

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

        # Keys to look for
        self.mode = mode
        if mode == 'ss_ds':
            self.keys = ['lon', 'lat', 'depth', 'dip', 
                            'width', 'length', 'strike', 
                            'strikeslip', 'dipslip']
        elif mode == 'mag_rake':
            self.keys = ['lon', 'lat', 'depth', 'dip', 
                            'width', 'length', 'strike', 
                            'magnitude', 'rake']
        else:
            raise ValueError("Invalid mode. Expected 'ss_ds' or 'mag_rake'.")

        # Initialize the fault objects
        self.faults = {f'fault_{i}': planarfault('mcmc fault {}'.format(i), utmzone=self.utmzone, 
                                                lon0=self.lon0, 
                                                lat0=self.lat0,
                                                ellps=self.ellps, 
                                                verbose=False) for i in range(num_faults)}
        self.faultnames = [f'fault_{i}' for i in range(len(self.faults))]

        # Load the configuration file
        if config_file is not None:
            self.load_config(config_file)
        
        # Set fixed parameters
        if hasattr(self, 'fixed_params') and self.fixed_params:
            self.fixed_params.update(fixed_params if fixed_params is not None else {})
        else:
            self.fixed_params = fixed_params if fixed_params is not None else {}

        # Initialize the index for each fault's parameters
        self.param_index = {}
        self.param_keys = {}
        self.total_params = 0
        index = 0
        for i in range(num_faults):
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

    def load_config(self, config_file):
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        self.bounds = config.get('bounds', {})
        self.initial = config.get('initial', {})
        self.sigmas = config.get('sigmas', {})
        self.fixed_params = config.get('fixed_params', {})
        self.ndatas = self.sigmas.get('ndatas', 0)
        self.dataFaults = config.get('dataFaults', None)
    
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
                initialSample.setdefault(key, pm_func.rvs())  # draw a sample for the initial sample

                # Save it
                if bound[0]!='Degenerate':
                    self.Priors.append(pm_func)
                    initSampleVec.append(initialSample[key])

        # Create a prior for the data set reference term
        # Works only for InSAR data yet
        if datas is not None:

            # Check 
            if type(datas) is not list:
                datas = [datas]
                
            # Iterate over the data
            for data in datas:
                
                # Get it
                assert data.name in bounds, \
                    'No bounds provided for prior for data {}'.format(data.name)
                bound = bounds[data.name]
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
            ndatas = self.sigmas['ndatas']
            self.sigmas_index = [len(self.Priors)+i for i in range(ndatas)]
            self.sigmas_keys = ['sigma_{}'.format(i) for i in range(ndatas)]
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
                initSampleVec.append(pm_func.rvs())
        else:
            self.sigmas_values = self.sigmas['values']

        # Save initial sample
        self.initSampleVec = initSampleVec
        self.initialSample = initialSample

        # All done
        return

    def setLikelihood(self, datas, vertical=True):
        '''
        Builds the data likelihood object from the list of geodetic data in datas.
    
        Args:   
            * datas         : csi geodetic data object (gps or insar) or list of csi geodetic objects. TODO: Add other types of data (opticorr)
    
        Kwargs:
            * vertical      : Use the verticals for GPS?
    
        Returns:
            * None
        '''
    
        # Build the prediction method
        # Initialize the object
        if type(datas) is not list:
            self.datas = [datas]
        else:
            self.datas = datas
    
        # List of likelihoods
        self.Likelihoods = []
    
        # Create a likelihood function for each of the data set
        for data in self.datas:
    
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
    
    def walk(self, nchains=200, chain_length=50, comm=None, filename='samples.h5',
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
        if self.dataFaults is None:
            self.dataFaults = dataFaults
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
            * model             : Can be 'mean', 'median', 'rand', 'MAP', an integer or a dictionary with the appropriate keys
    
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
        
        # Extract the sigmas
        if self.sigmas['update']:
            specs['sigmas'] = np.array(samples[self.sigmas_index])

        # Save the desired model 
        self.model_dict = specs
        self.model = samples

        # All done
        return faults
    
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
            for data in self.datas:

                # Build the green's functions
                fault.buildGFs(data, slipdir='sd', verbose=False, vertical=vertical)

        # Build the synthetics
        data.buildsynth(fault)

            # Check ref
        for data in self.datas:
            if '{}'.format(data.name) in self.keys:
                data.synth += self.model['{}'.format(data.name)]

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
        if self.sigmas['update']:
            print("Sigmas:")
            for ikey, key in enumerate(self.sigmas_keys):
                print(f"  {key}: {self.sigmas_index[ikey]}")

#EOF
