#!/usr/bin/python

# See http://maggotroot.blogspot.ch/2013/11/constrained-linear-least-squares-in.html for more info
'''
    A simple library to solve constrained linear least squares problems
    with sparse and dense matrices. Uses cvxopt library for
    optimization
'''

__author__ = 'Valeriy Vishnevskiy'
__email__ = 'valera.vishnevskiy@yandex.ru'
__version__ = '1.0'
__date__ = '22.11.2013'
__license__ = 'WTFPL'

import numpy as np
from cvxopt import solvers, matrix, spmatrix, mul
import itertools
from scipy import sparse


def scipy_sparse_to_spmatrix(A):
    coo = A.tocoo()
    SP = spmatrix(coo.data, coo.row.tolist(), coo.col.tolist())
    return SP

def spmatrix_sparse_to_scipy(A):
    data = np.array(A.V).squeeze()
    rows = np.array(A.I).squeeze()
    cols = np.array(A.J).squeeze()
    return sparse.coo_matrix( (data, (rows, cols)) )

def sparse_None_vstack(A1, A2):
    if A1 is None:
        return A2
    else:
        return sparse.vstack([A1, A2])

def numpy_None_vstack(A1, A2):
    if A1 is None:
        return A2
    else:
        # print(A1.shape, A2.shape)
        return np.vstack([A1, A2])

def numpy_None_concatenate(A1, A2):
    if A1 is None:
        return A2
    else:
        return np.concatenate([A1, A2])

def get_shape(A):
    if isinstance(C, spmatrix):
        return C.size
    else:
        return C.shape

def numpy_to_cvxopt_matrix(A):
    if A is None:
        return A
    if sparse.issparse(A):
        if isinstance(A, sparse.spmatrix):
            return scipy_sparse_to_spmatrix(A)
        else:
            return A
    else:
        if isinstance(A, np.ndarray):
            if A.ndim == 1:
                return matrix(A, (A.shape[0], 1), 'd')
            else:
                return matrix(A, A.shape, 'd')
        else:
            return A

def cvxopt_to_numpy_matrix(A):
    if A is None:
        return A
    if isinstance(A, spmatrix):
        return spmatrix_sparse_to_scipy(A)
    elif isinstance(A, matrix):
        return np.array(A).squeeze()
    else:
        return np.array(A).squeeze()


def lsqlin(C, d, reg=0, A=None, b=None, Aeq=None, beq=None, \
        lb=None, ub=None, x0=None, opts=None):
    '''
        Solve linear constrained l2-regularized least squares. Can
        handle both dense and sparse matrices. Matlab's lsqlin
        equivalent. It is actually wrapper around CVXOPT QP solver.
            min_x ||C*x  - d||^2_2 + reg * ||x||^2_2
            s.t.    A * x <= b
                    Aeq * x = beq
                    lb <= x <= ub
        Input arguments:
            C   is m x n dense or sparse matrix
            d   is n x 1 dense matrix
            reg is regularization parameter
            A   is p x n dense or sparse matrix
            b   is p x 1 dense matrix
            Aeq is q x n dense or sparse matrix
            beq is q x 1 dense matrix
            lb  is n x 1 matrix or scalar
            ub  is n x 1 matrix or scalar
        Output arguments:
            Return dictionary, the output of CVXOPT QP.
        Dont pass matlab-like empty lists to avoid setting parameters,
        just use None:
            lsqlin(C, d, 0.05, None, None, Aeq, beq) #Correct
            lsqlin(C, d, 0.05, [], [], Aeq, beq) #Wrong!
    '''
    sparse_case = False
    if sparse.issparse(A): #detects both np and cxopt sparse
        sparse_case = True
        #We need A to be scipy sparse, as I couldn't find how
        #CVXOPT spmatrix can be vstacked
        if isinstance(A, spmatrix):
            A = spmatrix_sparse_to_scipy(A)

    C =   numpy_to_cvxopt_matrix(C)
    d =   numpy_to_cvxopt_matrix(d)
    Q = C.T * C
    q = - d.T * C
    nvars = C.size[1]

    if reg > 0:
        if sparse_case:
            I = scipy_sparse_to_spmatrix(sparse.eye(nvars, nvars,\
                                        format='coo'))
        else:
            I = matrix(np.eye(nvars), (nvars, nvars), 'd')
        Q = Q + reg * I

    lb = cvxopt_to_numpy_matrix(lb)
    ub = cvxopt_to_numpy_matrix(ub)
    b  = cvxopt_to_numpy_matrix(b)
    #@ added by kfhe at 03/02/2023, avoid case where b to np.array(x)
    if b is not None and b.size == 1:
        b = np.array([b.item(0)])
    #@-----------------------------------------------------------@

    if lb is not None:  #Modify 'A' and 'b' to add lb inequalities
        if lb.size == 1:
            lb = np.repeat(lb, nvars)

        if sparse_case:
            lb_A = -sparse.eye(nvars, nvars, format='coo')
            A = sparse_None_vstack(A, lb_A)
        else:
            lb_A = -np.eye(nvars)
            A = numpy_None_vstack(A, lb_A)
        b = numpy_None_concatenate(b, -lb)
    if ub is not None:  #Modify 'A' and 'b' to add ub inequalities
        if ub.size == 1:
            ub = np.repeat(ub, nvars)
        if sparse_case:
            ub_A = sparse.eye(nvars, nvars, format='coo')
            A = sparse_None_vstack(A, ub_A)
        else:
            ub_A = np.eye(nvars)
            A = numpy_None_vstack(A, ub_A)
        b = numpy_None_concatenate(b, ub)

    #Convert data to CVXOPT format
    A =   numpy_to_cvxopt_matrix(A)
    Aeq = numpy_to_cvxopt_matrix(Aeq)
    b =   numpy_to_cvxopt_matrix(b)
    beq = numpy_to_cvxopt_matrix(beq)

    #Set up options
    if opts is not None:
        for k, v in opts.items():
            solvers.options[k] = v

    #Run CVXOPT.SQP solver
    sol = solvers.qp(Q, q.T, A, b, Aeq, beq, None, x0)
    return sol

def lsqnonneg(C, d, opts):
    '''
    Solves nonnegative linear least-squares problem:
    min_x ||C*x - d||_2^2,  where x >= 0
    '''
    return lsqlin(C, d, reg = 0, A = None, b = None, Aeq = None, \
                beq = None, lb = 0, ub = None, x0 = None, opts = opts)


if __name__ == '__main__':
    # simple Testing routines
    C = np.array(np.mat('''0.9501,0.7620,0.6153,0.4057;
    0.2311,0.4564,0.7919,0.9354;
    0.6068,0.0185,0.9218,0.9169;
    0.4859,0.8214,0.7382,0.4102;
    0.8912,0.4447,0.1762,0.8936'''))
    sC = sparse.coo_matrix(C)
    csC = scipy_sparse_to_spmatrix(sC)

    A = np.array(np.mat('''0.2027,0.2721,0.7467,0.4659;
    0.1987,0.1988,0.4450,0.4186;
    0.6037,0.0152,0.9318,0.8462'''))
    sA = sparse.coo_matrix(A)
    csA = scipy_sparse_to_spmatrix(sA)

    d = np.array([0.0578, 0.3528, 0.8131, 0.0098, 0.1388])
    md = matrix(d)

    b =  np.array([0.5251, 0.2026, 0.6721])
    mb = matrix(b)

    lb = np.array([-0.1] * 4)
    mlb = matrix(lb)
    mmlb = -0.1

    ub = np.array([2] * 4)
    mub = matrix(ub)
    mmub = 2

    #solvers.options[show_progress'] = False
    opts = {'show_progress': False}

    for iC in [C, sC, csC]:
        for iA in [A, sA, csA]:
            for iD in [d, md]:
                for ilb in [lb, mlb, mmlb]:
                    for iub in [ub, mub, mmub]:
                        for ib in [b, mb]:
                            ret = lsqlin(iC, iD, 0, iA, ib, None, None, ilb, iub, None, opts)
                            print(ret['x'].T)
    print('Should be [-1.00e-01 -1.00e-01  2.15e-01  3.50e-01]')

    #test lsqnonneg
    C = np.array([[0.0372, 0.2869], [0.6861, 0.7071], [0.6233, 0.6245], [0.6344, 0.6170]]);
    d = np.array([0.8587, 0.1781, 0.0747, 0.8405]);
    ret = lsqnonneg(C, d, {'show_progress': False})
    print(ret['x'].T)
    print('Should be [2.5e-07; 6.93e-01]')