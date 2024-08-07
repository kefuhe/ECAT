#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 11:00:48 2019

@author: duttar

Sequential Monte Carlo technique with parallelization using MPI

Changed by Kefeng He on 2023-11-16 for the parallelization of the SMC sampling

We introduce numba to accelerate the code, and use the MPI parallelization to speed up the code.
"""
import numpy as np
from numba import jit, njit
from multiprocessing import Pool
from joblib import Parallel, delayed
from numpy.random import RandomState
import time
from datetime import datetime
import h5py

# @njit
def deterministicR(inIndex, q):
    n_chains = inIndex.shape[0]
    parents = np.arange(n_chains)
    N_childs = np.zeros(n_chains, dtype=np.int64)

    cum_dist = np.cumsum(q)
    u = (parents + np.random.rand(1)) / n_chains

    j = 0
    for ui in u:
        while ui > cum_dist[j]:
            j += 1
        N_childs[j] += 1

    outindx = np.zeros(n_chains, dtype=np.int64)
    indx = 0
    for parent, N_child in zip(parents, N_childs):
        if N_child > 0:
            outindx[indx:indx + N_child] = parent
        indx += N_child

    return outindx

# @njit
def deterministicR_optimized(inIndex, q): 
    """
    Parameters: 
      * inIndex  : index of the posterior values  
      * q  : weight of the posterior values 
      
    Output: 
      * outindx  : index of the resampling  
    """
    
    n_chains = inIndex.shape[0]
    parents = np.arange(n_chains)

    cum_dist = np.cumsum(q)
    aux = np.random.rand(1)
    u = (parents + aux) / n_chains

    N_childs = np.zeros(n_chains, dtype=np.int64)
    j = np.searchsorted(cum_dist, u)
    np.add.at(N_childs, j, 1)
    # for index in j:
    #     N_childs[index] += 1

    outindx = np.repeat(parents, N_childs)

    return outindx


def AMH(X,target,covariance,mrun,beta,LB,UB):
    """
    Adaptive Metropolis algorithm
    scales the covariance matrix according to the acceptance rate 
    cov of proposal = (a+ bR)*sigma;  R = acceptance rate
    returns the last sample of the chain
    
    Parameters: 
     *  X : starting model 
     *  target : target distribution (is a function handle and calculates the log posterior)
     *  covariance : covariance of the proposal distribution 
     *  mrun : number of samples 
     *  beta : use the beta parameter of SMC sampling, otherwise 1 
     *  LB : lower bound of the model parameters
     *  UB : upper bound of the model parameters
    
    Outputs: 
     *  G : last sample of the chain 
     *  GP : log posterior value of the last sample 
     *  avg_acc : average acceptance rate
     
     written by : Rishabh Dutta (18 Mar 2019)
     Matlab version written on 12 Mar 2016
     (Don't forget to acknowledge)
    """
    
    Dims = covariance.shape[0]
    logpdf = target(X) 
    V = covariance
    best_P = logpdf * beta 
    P0 = logpdf * beta 
    
    # the following values are estimated empirically 
    a = 1/9
    b = 8/9
    
    sameind = np.where(np.equal(LB, UB))
    
    dimension = np.array([0.441, 0.352, 0.316, 0.285, 0.275, 
                          0.273, 0.270, 0.268, 0.267, 0.266, 0.265, 0.255])
    
    # set initial scaling factor
    s = a + b*dimension[min(Dims, 11)]
    
    U = np.log(np.random.rand(1,mrun))
    TH = np.zeros((Dims,mrun))
    THP = np.zeros((1,mrun))
    avg_acc = 0 
    factor = np.zeros((1,mrun))
    
    for i in range(mrun):
        X_new = np.random.multivariate_normal(X,s**2*V)
        X_new[sameind] = LB[sameind]
        
        ind1 = X_new < LB
        diff1 = LB[ind1] - X_new[ind1]
        X_new[ind1] = LB[ind1] + diff1 
        
        if avg_acc < 0.05: 
            X_new[ind1] = LB[ind1]
            
        ind2 = X_new > UB
        diff2 = X_new[ind2] - UB[ind2]
        X_new[ind2] = UB[ind2] - diff2
        
        if avg_acc < 0.05:
            X_new[ind2] = UB[ind2]
            
        P_new = beta * target(X_new)
        
        if P_new > best_P: 
            X = X_new
            best_P = P_new
            P0 = P_new
            acc_rate = 1 
        else:
            rho = P_new - P0 
            acc_rate = np.exp(np.min([0,rho]))
            if U[0,i] <= rho : 
                X = X_new
                P0 = P_new
                
        TH[:,i] = np.transpose(X) 
        THP[0,i] = P0 
        factor[0,i] = s**2
        avg_acc = avg_acc*(i)/(i+1) + acc_rate/(i+1) 
        s = a+ b*avg_acc
        
    G = TH[:,-1]
    GP = THP[0,-1]/beta
    
    return G, GP, avg_acc

@njit
def multivariate_normal(mean, L, size):
    Z = np.random.standard_normal((size, L.shape[0]))
    return (mean + np.dot(Z, L.T)) # [0]

@njit
def adjust_bounds(X_new, LB, UB, avg_acc):
    adjust = avg_acc >= 0.05
    ind1 = X_new < LB
    X_new[ind1] = np.add(LB[ind1], (LB[ind1] - X_new[ind1]) * adjust)

    ind2 = X_new > UB
    X_new[ind2] = np.subtract(UB[ind2], (X_new[ind2] - UB[ind2]) * adjust)

    return X_new

def run_amh(X, covariance_chol, mrun, beta, LB, UB, target, a=1.0/9.0, b=8.0/9.0):
    Dims = covariance_chol.shape[0]
    logpdf = target(X) 
    P = logpdf * beta 
    best_P = P
    P0 = P

    sameind = np.where(np.equal(LB, UB))
    dimension = np.array([0.441, 0.352, 0.316, 0.285, 0.275, 0.273, 0.270, 0.268, 
                          0.267, 0.266, 0.265, 0.255])
    s = a + b*dimension[np.minimum(Dims, 11)]

    L = covariance_chol  # Use pre-computed Cholesky decomposition

    U = np.log(np.random.rand(mrun))
    TH = np.zeros((Dims, mrun))
    THP = np.zeros(mrun)
    avg_acc = 0

    # Z ~ N(0, L)
    Z = multivariate_normal(np.zeros_like(X), L, mrun)  # Generate random numbers outside the loop

    for i in range(mrun):
        # X_new = multivariate_normal(X, s*L, 1)  # Update with s*L
        # X_new = s*Z + X, so X_new ~ N(X, s*L)
        X_new = X + s * Z[i]  # Add X and multiply by s inside the loop
        X_new[sameind] = LB[sameind]

        # 在run_amh函数中使用新的函数
        X_new = adjust_bounds(X_new, LB, UB, avg_acc)

        P_new = beta * target(X_new)

        if P_new > best_P: 
            X = X_new
            best_P = P_new
            P0 = P_new
            acc_rate = 1 
        else:
            rho = P_new - P0 
            # acc_rate = np.exp(np.minimum(0,rho))
            acc_rate = 1 if rho > 0 else np.exp(rho)
            if U[i] <= rho: 
                X = X_new
                P0 = P_new

        TH[:, i] = X
        THP[i] = best_P
        inv_i_plus_1 = 1.0 / (i + 1)
        avg_acc = avg_acc * i * inv_i_plus_1 + acc_rate * inv_i_plus_1
        s = a + b*avg_acc

    return TH, THP, avg_acc

def AMH_optimized_jit(X, target, covariance_chol, mrun, beta, LB, UB, a=1.0/9.0, b=8.0/9.0):
    TH, THP, avg_acc = run_amh(X, covariance_chol, mrun, beta, LB, UB, target, a, b)
    G = TH[:, -1]
    GP = THP[-1] / beta
    return G, GP, avg_acc


@njit
def calculate_covariance(smpldiff, probwght):
    dims = smpldiff.shape[1]
    covariance = np.zeros((dims, dims))
    for i in range(smpldiff.shape[0]):
        covariance += probwght[i] * np.outer(smpldiff[i], smpldiff[i])
    return covariance


@njit
def calculate_weights(logpst, beta1, beta2):
    logwght = (beta1 - beta2) * logpst
    wght = np.exp(logwght)
    probwght = wght / np.sum(wght)
    return probwght


# %%
class SMCclass:
    """
    Generates samples of the 'target' posterior PDF using SMC sampling. Also called Adapative Transitional Metropolis
    Importance (sampling) P abbreviated as ATMIP  
    """
    def __init__(self, opt, samples, NT1, NT2, verbose=True):
        """
        Parameters: 
            opt : named tuple 
                - opt.target (lamda function of the posterior)
                - opt.UB (upper bound of parameters)
                - opt.LB (lower bound of parameters)
                - opt.N (number of Markov chains at each stage)
                - opt.Neff (Chain length of the MCMC sampling) 
                
            samples: named tuple
                - samples.allsamples (samples at each stage)
                - samples.postval (log posterior value of samples)
                - samples.beta (array of beta values)
                - samples.stage (array of stages)
                - samples.covsmpl (model covariance at each stage)
                - samples.resmpl (resampled model at each stage)
                
            NT1: create opt object
            NT2: create samples object 
            
        written by: Rishabh Dutta, Dec 12 2018
        (Don't forget to acknowledge)
        
        """
        self.verbose = verbose
        self.opt = opt  
        self.samples = samples
        self.NT1 = NT1
        self.NT2 = NT2
            
    def initialize(self):
        if self.verbose:
            print ("-----------------------------------------------------------------------------------------------")
            print ("-----------------------------------------------------------------------------------------------")
            print(f'Initializing ATMIP with {self.opt.N :8d} Markov chains and {self.opt.Neff :8d} chain length.')
                    
    def prior_samples(self):
        '''
        determines the prior posterior values 
        the prior samples are estimated from lower and upper bounds
        
        Output : samples (NT2 object with estimated posterior values)
        '''
        numpars = self.opt.LB.shape[0]
        diffbnd = self.opt.UB - self.opt.LB
        diffbndN = np.tile(diffbnd,(self.opt.N,1))
        LBN = np.tile(self.opt.LB,(self.opt.N,1))
        
        sampzero = LBN +  np.random.rand(self.opt.N,numpars) * diffbndN
        beta = np.array([0]) 
        stage = np.array([1]) 
        
        postval = np.zeros([self.opt.N,1])
        for i in range(self.opt.N):
            samp0 = sampzero[i,:]
            logpost = self.opt.target(samp0)
            postval[i] = logpost
            
        samples = self.NT2(sampzero, postval, beta, stage, None, None)
        return samples

    def prior_samples_vectorize(self):
        '''
        determines the prior posterior values 
        the prior samples are estimated from lower and upper bounds
        
        Output : samples (NT2 object with estimated posterior values)
        '''
        numpars = self.opt.LB.shape[0]
        diffbnd = self.opt.UB - self.opt.LB
        diffbndN = np.tile(diffbnd,(self.opt.N,1))
        LBN = np.tile(self.opt.LB,(self.opt.N,1))
        
        sampzero = LBN +  np.random.rand(self.opt.N,numpars) * diffbndN
        beta = np.array([0]) 
        stage = np.array([1]) 
        
        # 使用NumPy的向量化操作替换for循环
        logpost = np.apply_along_axis(self.opt.target, 1, sampzero)
        postval = logpost.reshape(-1, 1)
            
        samples = self.NT2(sampzero, postval, beta, stage, None, None)
        return samples
          
    
    def find_beta(self): 
        """
        Calculates the beta parameter for the next stage
        """
        beta1 = self.samples.beta[-1]       #prev_beta
        beta2 = self.samples.beta[-1]       #prev_beta
        max_post = np.max(self.samples.postval) 
        logpst = self.samples.postval - max_post
        beta = beta1+.5
    
        if beta>1:
            beta = 1
            #logwght = beta.*logpst
            #wght = np.exp(logwght)
    
        refcov = 1 
    
        # Binary search to find the beta parameter
        while beta - beta1 > 1e-6:
            curr_beta = (beta+beta1)/2
            diffbeta = beta-beta1
            logwght = diffbeta*logpst
            wght = np.exp(logwght)
            covwght = np.std(wght)/np.mean(wght)
            if covwght > refcov:
                beta = curr_beta
            else:
                beta1 = curr_beta
            
        betanew = np.min(np.array([1,beta]))
        betaarray = np.append(self.samples.beta,betanew)
        newstage = np.arange(1,self.samples.stage[-1]+2)
        samples = self.NT2(self.samples.allsamples, self.samples.postval, \
                           betaarray, newstage, self.samples.covsmpl, \
                           self.samples.resmpl)
    
        return samples
    
    def find_beta_optimized(self): 
        """
        Calculates the beta parameter for the next stage
        """
        beta1 = self.samples.beta[-1]       #prev_beta
        beta2 = self.samples.beta[-1]       #prev_beta
        max_post = np.max(self.samples.postval) 
        logpst = self.samples.postval - max_post
        beta = beta1+.5

        if beta>1:
            beta = 1

        refcov = 1 

        while beta - beta1 > 1e-6:
            curr_beta = (beta+beta1)/2
            diffbeta = beta-beta1
            logwght = diffbeta*logpst
            wght = np.exp(logwght)

            # Use online algorithm to calculate standard deviation and mean
            mean = 0
            M2 = 0
            for i in range(len(wght)):
                delta = wght[i] - mean
                mean += delta / (i + 1)
                delta2 = wght[i] - mean
                M2 += delta * delta2
            var = M2 / (len(wght) - 1)
            std_dev = np.sqrt(var)
            covwght = std_dev / mean

            if covwght > refcov:
                beta = curr_beta
            else:
                beta1 = curr_beta

        betanew = np.min(np.array([1,beta]))
        betaarray = np.append(self.samples.beta,betanew)
        newstage = np.arange(1,self.samples.stage[-1]+2)
        samples = self.NT2(self.samples.allsamples, self.samples.postval, \
                           betaarray, newstage, self.samples.covsmpl, \
                           self.samples.resmpl)

        return samples
    
    def resample_stage(self):
        '''
        Resamples the model samples at a certain stage 
        Uses Kitagawa's deterministic resampling algorithm
        '''
        
        # calculate the weight for model samples
        logpst = self.samples.postval - np.max(self.samples.postval)
        logwght = (self.samples.beta[-1] - self.samples.beta[-2])* logpst
        wght = np.exp(logwght)
        
        probwght = wght/np.sum(wght)
        inind = np.arange(0,self.opt.N)
        
        outind = deterministicR_optimized(inind, probwght) # deterministicR_optimized
        newsmpl = self.samples.allsamples[outind,:]
        
        samples = self.NT2(self.samples.allsamples, self.samples.postval, \
                           self.samples.beta, self.samples.stage, \
                           self.samples.covsmpl, newsmpl)
        
        return samples

    def resample_stage_optimized(self):
        '''
        Resamples the model samples at a certain stage 
        Uses Kitagawa's deterministic resampling algorithm
        '''
        
        # calculate the weight for model samples
        logpst = self.samples.postval - np.max(self.samples.postval)
        probwght = calculate_weights(logpst, self.samples.beta[-1], self.samples.beta[-2])
        inind = np.arange(0,self.opt.N)
        
        outind = deterministicR_optimized(inind, probwght) # deterministicR_optimized
        newsmpl = self.samples.allsamples[outind,:]
        
        samples = self.NT2(self.samples.allsamples, self.samples.postval, \
                           self.samples.beta, self.samples.stage, \
                           self.samples.covsmpl, newsmpl)
        
        return samples
        
    def make_covariance(self):
        '''
        make the model covariance using the weights and samples from previous 
        stage
        '''
        # calculate the weight for model samples
        
        dims = self.samples.allsamples.shape[1]
        logpst = self.samples.postval - np.max(self.samples.postval)
        logwght = (self.samples.beta[-1] - self.samples.beta[-2])* logpst
        wght = np.exp(logwght)
        
        probwght = wght/np.sum(wght)
        weightmat = np.tile(probwght,(1,dims))
        multmat = weightmat * self.samples.allsamples
        
        # calculate the mean samples
        meansmpl = multmat.sum(axis=0, dtype='float')
        
        # calculate the model covariance
        covariance = np.matrix(np.zeros((dims,dims), dtype='float'))
        for i in range(self.opt.N):
            par = self.samples.allsamples[i,:]
            smpldiff = np.matrix(par - meansmpl)
            smpdsq = np.matmul(np.transpose(smpldiff),smpldiff)
            covint = np.multiply(probwght[i], smpdsq)
            covariance += covint
            
        samples = self.NT2(self.samples.allsamples, self.samples.postval, \
                           self.samples.beta, self.samples.stage, \
                           covariance, self.samples.resmpl)
        return samples

    def make_covariance_optimized(self, epsilon = 1e-6):
        '''
        make the model covariance using the weights and samples from previous 
        stage
        '''
        # calculate the weight for model samples
        
        dims = self.samples.allsamples.shape[1]
        logpst = self.samples.postval - np.max(self.samples.postval)
        logwght = (self.samples.beta[-1] - self.samples.beta[-2])* logpst
        wght = np.exp(logwght)
        
        probwght = wght/np.sum(wght)
        weightmat = np.tile(probwght,(1,dims))
        multmat = weightmat * self.samples.allsamples
        
        # calculate the mean samples
        meansmpl = multmat.sum(axis=0, dtype='float')
        
        # calculate the model covariance
        smpldiff = self.samples.allsamples - meansmpl  # Calculate smpldiff outside the loop
        # Way 1: Use numpy.einsum to calculate smpdsq
        smpdsq = np.einsum('ij,ik->ijk', smpldiff, smpldiff)  # Use numpy.einsum to calculate smpdsq
        covariance = np.sum(probwght[:, None] * smpdsq, axis=0)  # Use numpy broadcasting and sum to calculate covariance
        # Way 2: Use numpy.outer to calculate smpdsq
        # covariance = np.zeros((dims, dims))
        # for i in range(smpldiff.shape[0]):
        #     covariance += probwght[i] * np.outer(smpldiff[i], smpldiff[i])

        # 在计算协方差矩阵后添加一个小的正数到对角线上
        epsilon = epsilon
        covariance += epsilon * np.eye(dims)

        samples = self.NT2(self.samples.allsamples, self.samples.postval, \
                        self.samples.beta, self.samples.stage, \
                        covariance, self.samples.resmpl)
        return samples

    def make_covariance_optimized_jit(self, epsilon = 1e-6):
        '''
        make the model covariance using the weights and samples from previous 
        stage
        '''
        # calculate the weight for model samples
        
        dims = self.samples.allsamples.shape[1]
        logpst = self.samples.postval - np.max(self.samples.postval)
        logwght = (self.samples.beta[-1] - self.samples.beta[-2])* logpst
        wght = np.exp(logwght)
        
        probwght = wght/np.sum(wght)
        weightmat = np.tile(probwght,(1,dims))
        multmat = weightmat * self.samples.allsamples
        
        # calculate the mean samples
        meansmpl = multmat.sum(axis=0, dtype='float')
        
        # calculate the model covariance
        smpldiff = self.samples.allsamples - meansmpl  # Calculate smpldiff outside the loop
        covariance = calculate_covariance(smpldiff, probwght)

        # 在计算协方差矩阵后添加一个小的正数到对角线上
        epsilon = epsilon
        covariance += epsilon * np.eye(dims)

        samples = self.NT2(self.samples.allsamples, self.samples.postval, \
                        self.samples.beta, self.samples.stage, \
                        covariance, self.samples.resmpl)
        return samples
        
    def MCMC_samples(self):
        """
        Nothing
        """
        dims = self.samples.allsamples.shape[1]
        
        mhsmpl = np.zeros([self.opt.N,dims])
        mhpost = np.zeros([self.opt.N,1])
        for i in range(self.opt.N):
            start = self.samples.resmpl[i,:]
            G, GP, acc = AMH_optimized_jit(start, self.opt.target, self.samples.covsmpl, \
                               self.opt.Neff, self.samples.beta[-1], \
                               self.opt.LB, self.opt.UB)
            mhsmpl[i,:] = np.transpose(G)
            mhpost[i] = GP 
            
        samples = self.NT2(mhsmpl, mhpost, self.samples.beta, \
                           self.samples.stage, self.samples.covsmpl, \
                           self.samples.resmpl)
        return samples

    def MCMC_samples_parallel_mpi(self, comm=None, a=1.0/9.0, b=8.0/9.0):
        comm = comm
        rank = comm.Get_rank()
        size = comm.Get_size()

        dims = self.samples.allsamples.shape[1]
        mhsmpl = np.zeros([self.opt.N,dims])
        mhpost = np.zeros([self.opt.N,1])

        # Calculate Cholesky decomposition at rank 0 and broadcast to all ranks
        # covsmpl_chol = None
        # if rank == 0:
        covsmpl_chol = np.linalg.cholesky(self.samples.covsmpl)
        # covsmpl_chol = comm.bcast(covsmpl_chol, root=0)

        def process_sample(i):
            start = self.samples.resmpl[i,:]
            G, GP, _ = AMH_optimized_jit(start, self.opt.target, covsmpl_chol, \
                            self.opt.Neff, self.samples.beta[-1], \
                            self.opt.LB, self.opt.UB, a, b)
            return np.transpose(G), GP

        results = [process_sample(i) for i in range(rank, self.opt.N, size)]

        comm.Barrier()
        gathered_results = comm.gather(results, root=0)

        if rank == 0:
            gathered_results = [item for sublist in gathered_results for item in sublist]
            for i, (G, GP) in enumerate(gathered_results):
                mhsmpl[i,:] = G
                mhpost[i] = GP

            samples = self.NT2(mhsmpl, mhpost, self.samples.beta, \
                            self.samples.stage, self.samples.covsmpl, \
                            self.samples.resmpl)
        else:
            samples = None

        comm.Barrier()
        samples = comm.bcast(samples, root=0)

        return samples
                

# %%
def SMC_samples(opt,samples, NT1, NT2):
    '''
    Sequential Monte Carlo technique
    < a subset of CATMIP by Sarah Minson>
    The method samples the target distribution through several stages (called 
    transitioning of simulated annealing). At each stage the samples corresponds
    to the intermediate PDF between the prior PDF and final target PDF. 
    
    After samples generated at each stage, the beta parameter is generated for
    the next stage. At the next stage, resampling is performed. Then MCMC 
    sampling (adpative Metropolis chains) is resumed from each resampled model. 
    The weigted covariance is estimated using the weights (calculated from 
    posterior values) and samples from previous stage. This procedure is conti-
    nued until beta parameter is 1. 
    
    syntax: output = ATMIP(opt)
    
    Inputs: 
    
        opt : named tuple 
            - opt.target (lamda function of the posterior)
            - opt.UB (upper bound of parameters)
            - opt.LB (lower bound of parameters)
            - opt.N (number of Markov chains at each stage)
            - opt.Neff (Chain length of the MCMC sampling)
            
        samples: named tuple
            - samples.allsamples (samples at an intermediate stage)
            - samples.beta (beta at that stage)
            - samples.postval (posterior values)
            - samples.stage (stage number)
            - samples.covsmpl (model covariance matrix used for MCMC sampling)
    
        NT1 - named tuple structure for opt
        NT2 - named tuple structure for samples
        
    Outputs: 
        
        samples : named tuple
            - samples.allsamples (final samples at the last stage)
            - samples.postval (log posterior values of the final samples)
            - samples.stages (array of all stages)
            - samples.beta (array of beta values)
            - samples.covsmpl (model covariance at final stage)
            - samples.resmpl (resampled model samples at final stage)
            
    written by: Rishabh Dutta, Mar 25 2019
    (Don't forget to acknowledge)
    '''
    current = SMCclass(opt, samples, NT1, NT2)
    current.initialize()            # prints the initialization
    
    if samples.allsamples is None:  # generates prior samples and calculates 
                                    # their posterior values
        print('------Calculating the prior posterior values at stage 1-----')
        current = SMCclass(opt, samples, NT1, NT2)
        samples = current.prior_samples_parallel_joblib(n_jobs=-1)
        
    while samples.beta[-1] != 1:
        current = SMCclass(opt, samples, NT1, NT2)
        samples = current.find_beta()            # calculates beta at next stage 
        
        # at next stage here -------------------------------------
        current = SMCclass(opt, samples, NT1, NT2)
        samples = current.resample_stage()       # resample the model samples 
        
        # make the model covariance 
        current = SMCclass(opt, samples, NT1, NT2)
        samples = current.make_covariance_optimized()       
        
        # use the resampled model samples as starting point for MCMC sampling 
        # we use adaptive Metropolis sampling 
        # adaptive proposal is generated using model covariance 
        print(f'Starting metropolis chains at stage = {samples.stage[-1] :3d} and beta = {samples.beta[-1] :.6f}.')
        current = SMCclass(opt, samples, NT1, NT2)
        samples = current.MCMC_samples_parallel_joblib(n_jobs=-1)

    
    return samples


def SMC_samples_parallel_mpi(opt,samples, NT1, NT2, comm=None, save_at_final_stage=True,
                             save_interval=1, save_at_interval=False,
                            covariance_epsilon = 1e-6, amh_a=1.0/9.0, amh_b=8.0/9.0):
    '''
    Sequential Monte Carlo technique
    < a subset of CATMIP by Sarah Minson>
    The method samples the target distribution through several stages (called 
    transitioning of simulated annealing). At each stage the samples corresponds
    to the intermediate PDF between the prior PDF and final target PDF. 
    
    After samples generated at each stage, the beta parameter is generated for
    the next stage. At the next stage, resampling is performed. Then MCMC 
    sampling (adpative Metropolis chains) is resumed from each resampled model. 
    The weigted covariance is estimated using the weights (calculated from 
    posterior values) and samples from previous stage. This procedure is conti-
    nued until beta parameter is 1. 
    
    syntax: output = ATMIP(opt)
    
    Inputs: 
    
        opt : named tuple 
            - opt.target (lamda function of the posterior)
            - opt.UB (upper bound of parameters)
            - opt.LB (lower bound of parameters)
            - opt.N (number of Markov chains at each stage)
            - opt.Neff (Chain length of the MCMC sampling)
            
        samples: named tuple
            - samples.allsamples (samples at an intermediate stage)
            - samples.beta (beta at that stage)
            - samples.postval (posterior values)
            - samples.stage (stage number)
            - samples.covsmpl (model covariance matrix used for MCMC sampling)
    
        NT1 - named tuple structure for opt
        NT2 - named tuple structure for samples
        
    Outputs: 
        
        samples : named tuple
            - samples.allsamples (final samples at the last stage)
            - samples.postval (log posterior values of the final samples)
            - samples.stages (array of all stages)
            - samples.beta (array of beta values)
            - samples.covsmpl (model covariance at final stage)
            - samples.resmpl (resampled model samples at final stage)
            
    written by: Rishabh Dutta, Mar 25 2019
    (Don't forget to acknowledge)
    '''
    comm = comm
    rank = comm.Get_rank()
    size = comm.Get_size()
    # print(size, rank)

    current = SMCclass(opt, samples, NT1, NT2)
    if  rank == 0:
        current.initialize()           

    if samples.allsamples is None:  
        if rank == 0:
            print('------Calculating the prior posterior values at stage 1-----', flush=True)
            current = SMCclass(opt, samples, NT1, NT2)
            samples = current.prior_samples_vectorize()
            start_time = time.time()
        else:
            samples = None
    else:
        if rank == 0:
            start_time = time.time()
        
    # 等待所有进程完成计算
    comm.Barrier()
    # 广播samples到所有的进程
    samples = comm.bcast(samples, root=0)
        
    while samples.beta[-1] != 1:
        if rank == 0:
            current = SMCclass(opt, samples, NT1, NT2)
            samples = current.find_beta_optimized()
        
            current = SMCclass(opt, samples, NT1, NT2)
            samples = current.resample_stage() # resample_stage_optimized
        
            current = SMCclass(opt, samples, NT1, NT2)
            samples = current.make_covariance_optimized(epsilon=covariance_epsilon)
        
            print(f'Starting metropolis chains at stage = {samples.stage[-1] :3d} and beta = {samples.beta[-1] :.6f}.', flush=True)

            end_time = time.time()

            # 计算并打印执行时间
            execution_time = end_time - start_time
            current_time = datetime.now().strftime("%y-%m-%d %H:%M:%S")
            print(f'The while loop took {execution_time:.6f} seconds to execute. Current time: {current_time}', flush=True)

            if save_at_interval and samples.stage[-1] % save_interval == 0:
                with h5py.File(f'samples_stage_{samples.stage[-1]}.h5', 'w') as f:
                    for key, value in samples._asdict().items():
                        f.create_dataset(key, data=value)
        else:
            samples = None


        # 等待所有进程完成计算
        comm.Barrier()
        # 广播samples到所有的进程
        samples = comm.bcast(samples, root=0)

        current = SMCclass(opt, samples, NT1, NT2)
        samples = current.MCMC_samples_parallel_mpi(comm=comm, a=amh_a, b=amh_b)

        # 等待所有进程完成计算
        comm.Barrier()
        # 广播samples到所有的进程
        samples = comm.bcast(samples, root=0)

    if rank == 0 and save_at_final_stage:
        with h5py.File('samples_final.h5', 'w') as f:
            for key, value in samples._asdict().items():
                f.create_dataset(key, data=value)

    return samples