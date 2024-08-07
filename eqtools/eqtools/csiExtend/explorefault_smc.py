'''
A class that searches for the best fault to fit some geodetic data.
This class is made for a simple planar fault geometry.
It is close to what R. Grandin has implemented but with a MCMC approach
Grandin's approach will be coded in another class.

Author:
R. Jolivet 2017

Modifications:
Changed by Kefeng He on 2023-11-16 for the purpose of exploring single fault model with SMC method.
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
from csi.SourceInv import SourceInv
from csi import planarfault
from .SMC_MPI import SMC_samples_parallel_mpi
from numba import njit
from collections import namedtuple

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

    Returns:
        * None
    '''

    def __init__(self, name, mode='ss_ds', utmzone=None, 
                    ellps='WGS84', lon0=None, lat0=None, 
                    verbose=True):

        # Initialize the fault
        if verbose:
            print ("---------------------------------")
            print ("---------------------------------")
            print ("Initializing fault exploration {}".format(name))
        self.verbose = verbose

        # Base class init
        super(explorefault,self).__init__(name, utmzone=utmzone, 
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

        # Initialize the fault object
        self.fault = planarfault('mcmc fault', utmzone=self.utmzone, 
                                                lon0=self.lon0, 
                                                lat0=self.lat0,
                                                ellps=self.ellps, 
                                                verbose=False)

        # All done
        return

    def setPriors(self, bounds, datas=None, initialSample=None):
        '''
        Initializes the prior likelihood functions.

        Args:
            * bounds        : Bounds is a dictionary that holds the following keys. 
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

                              One bound should be a list with the name of a pymc distribution as first element. The following elements will be passed on to the function.
                              example:  bounds[0] = ('Normal', 0., 2.) will give a Normal distribution centered on 0. with a 2. standard deviation.

        Kwargs:
            * datas         : Data sets that will be used. This is in case bounds has tuples or floats for reference of an InSAR data set

            * initialSample : An array the size of the list of bounds default is None and will be randomly set from the prior PDFs

        Returns:
            * None
        '''

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

        # What do we sample?
        self.sampledKeys = {}
        isample = 0

        # Iterate over the keys
        for key in self.keys:

            # Check the key has been provided
            assert key in bounds, '{} not defined in the input dictionary'

            # Get the values
            bound = bounds[key]

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
                self.sampledKeys[key] = isample
                isample += 1

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
                    self.sampledKeys[key] = isample
                    isample += 1
                    self.keys.append(key)
                data.refnumber = len(self.Priors)-1

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
            ilike = [dobs, Cd_inv, Cd_inv, logCd_det]
            # Save the likelihood function
            self.Likelihoods.append([data]+ [il.astype(np.float64) for il in ilike] + [vertical])
    
        # All done 
        return

    def Predict(self, theta, data, vertical=True):
        '''
        Calculates a prediction of the measurement from the theta vector

        Args:
            * theta     : model parameters [lon, lat, depth, dip, width, length, strike, strikeslip, dipslip]
                          or [lon, lat, depth, dip, width, length, strike, magnitude, rake] depending on self.mode
            * data      : Data to test upon

        Kwargs:
            * vertical  : True/False

        Returns:
            * None
        '''
        # Take the values in theta and distribute
        lon = self._getFromTheta(theta, 'lon')
        lat = self._getFromTheta(theta, 'lat') 
        depth = self._getFromTheta(theta, 'depth')
        dip = self._getFromTheta(theta, 'dip')
        width = self._getFromTheta(theta, 'width')
        length = self._getFromTheta(theta, 'length')
        strike = self._getFromTheta(theta, 'strike')

        if self.mode == 'ss_ds':
            strikeslip = self._getFromTheta(theta, 'strikeslip')
            dipslip = self._getFromTheta(theta, 'dipslip')
        elif self.mode == 'mag_rake':
            magnitude = self._getFromTheta(theta, 'magnitude')
            rake = self._getFromTheta(theta, 'rake')
            strikeslip = magnitude*np.cos(np.radians(rake))
            dipslip = magnitude*np.sin(np.radians(rake))

        if hasattr(data, 'refnumber'):
            reference = theta[data.refnumber]
        else:
            reference = 0.

        # Get the fault
        fault = self.fault

        # Build a planar fault
        fault.buildPatches(lon, lat, depth, strike, dip, 
                           length, width, 1, 1, verbose=False)

        # Build the green's functions
        fault.buildGFs(data, vertical=vertical, slipdir='sd', verbose=False)

        # Set slip 
        fault.slip[:,0] = strikeslip
        fault.slip[:,1] = dipslip

        # Build the synthetics
        data.buildsynth(fault)

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
    
    def make_target(self):
        # Extract lb and ub from self.Priors
        self.lb = np.array([prior.args[0] for prior in self.Priors])
        self.ub = np.array([prior.args[0] + prior.args[1] for prior in self.Priors])
    
        def target(samples):
            log_prior = compute_log_prior(samples, self.lb, self.ub)
            if log_prior == -np.inf:
                return -np.inf
            else:
                log_likelihood = 0
                for data, dobs, Cd_inv, Cd_det, logCd_det, vertical in self.Likelihoods:
                    simulations = self.Predict(samples, data, vertical=vertical)
                    log_likelihood += compute_data_log_likelihood(simulations, dobs, Cd_inv, logCd_det)
                return log_prior + log_likelihood
        return target
    
    def walk(self, nchains=200, chain_length=50, comm=None, filename='samples.h5',
             save_every=1, save_at_interval=True, save_at_final=True,
             covariance_epsilon = 1e-6, amh_a=1.0/9.0, amh_b=8.0/9.0):
        '''
        March the SMC.
    
        Kwargs:
            * nchains           : Number of Markov Chains
            * chain_length      : Length of each chain
            * print_samples     : Whether to print the samples
            * filename          : The name of the HDF5 file to save the samples
    
        Returns:
            * None
        '''
    
        # Create a target function
        target = self.make_target()
    
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

    def returnModel(self, model='mean'):
        '''
        Returns a fault corresponding to the desired model.

        Kwargs:
            * model             : Can be 'mean', 'median',  'rand', an integer or a dictionary with the appropriate keys

        Returns:
            * fault instance
        '''

        # Create a dictionary
        specs = {}

        # Iterate over the keys
        for key in self.sampledKeys:
            
            # Get index
            ikey = self.sampledKeys[key]
        
            # Get it 
            if model=='mean':
                value = self.sampler['allsamples'][:,ikey].mean()
            elif model=='median':
                value = self.sampler['allsamples'][:,ikey].median()
            elif model=='std':                     
                value = self.sampler['allsamples'][:,ikey].std()
            else: 
                if type(model) is int:
                    assert type(model) is int, 'Model type unknown: {}'.format(model)
                    value = self.sampler['allsamples'][model,ikey]
                elif type(model) is dict:
                    value = model[key]

            # Set it
            specs[key] = value

        # Iterate over the others
        for key in self.keys:
            if key not in specs:
                specs[key] = self.initialSample[key]

        # Create a fault
        fault = planarfault('{} model'.format(model), 
                            utmzone=self.utmzone, 
                            lon0=self.lon0, 
                            lat0=self.lat0,
                            ellps=self.ellps, 
                            verbose=False)
        fault.buildPatches(specs['lon'], specs['lat'], 
                           specs['depth'], specs['strike'],
                           specs['dip'], specs['length'],
                           specs['width'], 1, 1, verbose=False)
        
        # Set slip values
        if self.mode == 'mag_rake':
            fault.slip[:,0] = specs['magnitude']*np.cos(np.radians(specs['rake']))
            fault.slip[:,1] = specs['magnitude']*np.sin(np.radians(specs['rake']))
        elif self.mode == 'ss_ds':
            fault.slip[:,0] = specs['strikeslip']
            fault.slip[:,1] = specs['dipslip']

        # Save the desired model 
        self.model = specs

        # All done
        return fault
    
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
        fault = self.returnModel(model=model)

        # Build predictions
        for data in self.datas:

            # Build the green's functions
            fault.buildGFs(data, slipdir='sd', verbose=False, vertical=vertical)

            # Build the synthetics
            data.buildsynth(fault)

            # Check ref
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
                        style='white', fill=True, scatter=False, scatter_size=15):
        import seaborn as sns
        import pandas as pd
        import matplotlib.pyplot as plt
        import numpy as np
    
        # Get the SMC chains
        trace = self.sampler['allsamples']
        keys = self.keys
    
        # Convert the SMC chains to a DataFrame
        df = pd.DataFrame(trace, columns=keys)
    
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

    def _getFromTheta(self, theta, string):
        '''
        Returns the value from the set of sampled and unsampled pdfs
        '''

        # Try to get the value
        if string in self.sampledKeys:
            return theta[self.sampledKeys[string]]
        else:
            return self.initialSample[string]

#EOF
