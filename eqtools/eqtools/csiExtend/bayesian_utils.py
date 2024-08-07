import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import splu
from scipy.linalg import cholesky
from numba import njit

## Cholesky一般计算行列式都比较快，但是精度相对差一点儿，但比np.linalg.det要快数百倍之多
## 稀疏矩阵的行列式计算，还是稀疏矩阵的lu分解比较快，比cholesky分解快数十倍之多，且此时精度相对较高

def log_determinant(A, inverse=False):
    """
    Calculates the natural logarithm of a determinant of the given matrix '
    according to the properties of a triangular matrix.

    Parameters
    ----------
    A : n x n :class:`numpy.ndarray`
    inverse : boolean
        If true calculates the log determinant of the inverse of the colesky
        decomposition, which is equivalent to taking the determinant of the
        inverse of the matrix.

        L.T* L = R           inverse=False
        L-1*(L-1)T = R-1     inverse=True

    Returns
    -------
    float logarithm of the determinant of the input Matrix A
    """

    chol = cholesky(A, lower=True)
    if inverse:
        chol = np.linalg.inv(chol)
    return np.log(np.diag(chol)).sum() * 2.0

def det_of_transpose_times_lu(matrix: csr_matrix) -> float:
    matrix_t_times = (matrix.transpose() @ matrix).tocsc()
    try:
        lu = splu(matrix_t_times)
        det = np.abs(np.prod(lu.U.diagonal()))
    except RuntimeError:
        det = 0
    return det

def det_of_laplace_smooth_lu(matrix: csr_matrix) -> float:
    if matrix.shape[1] > matrix.shape[0]:
        return 0
    else:
        return det_of_transpose_times_lu(matrix)

def det_of_transpose_times_cholesky(matrix: csr_matrix, epsilon=1e-15) -> float:
    matrix_t_times = (matrix.transpose() @ matrix).tocsc()
    matrix_t_times += epsilon * np.eye(matrix_t_times.shape[0])
    try:
        c = cholesky(matrix_t_times)
        det = np.prod(np.diag(c)) ** 2
    except np.linalg.LinAlgError:
        det = 0
    return det

def det_of_laplace_smooth_cholesky(matrix: csr_matrix, epsilon=1e-15) -> float:
    if matrix.shape[1] > matrix.shape[0]:
        return 0
    else:
        n = matrix.shape[0] // 2
        lap = matrix[:n, :n]
        det = det_of_transpose_times_cholesky(lap, epsilon)**2
        return det

@njit
def logpdf_multivariate_normal(x, mean, inv_cov, logdet):
    # size = len(x)
    # norm_const = -0.5 * (size * np.log(2*np.pi) + logdet)
    norm_const = -0.5 * logdet
    x_mu = np.subtract(x, mean)
    solution = np.dot(inv_cov, x_mu)
    result = -0.5 * np.dot(x_mu, solution) + norm_const
    return result


class Moment:
    def __init__(self, Mw_mean=6.0, Mw_sigma=1.0, shear_modulus=30, patch_areas=None, slip_positions=None):
        '''
        Parameters:
        Mw_mean: float
            Mean of the moment magnitude distribution
        Mw_sigma: float
            Standard deviation of the moment magnitude distribution
        shear_modulus: float
            Shear modulus of the fault, unit: GPa
        patch_areas: list of floats
            Areas of the patches, unit: km^2
        slip_positions: list of floats
            Positions of the patches
        '''
        self.shear_modulus = shear_modulus
        self.patch_areas = np.array(patch_areas)*1e6
        self.slip_positions = np.array(slip_positions, dtype=int)
        self.Mw_mean = Mw_mean
        self.Mw_sigma = Mw_sigma

    def initializeSample(self, rng=None, lb=None, ub=None):
        if rng is None:
            rng = np.random.default_rng()
        self.rng = rng
        self.Mw = rng.normal(self.Mw_mean, self.Mw_sigma)
        self.moment = 10 ** (1.5 * self.Mw + 9.1)
        self.patch_moment = self.moment / len(self.patch_areas)
        self.patch_slip = self.patch_moment / self.shear_modulus
        if lb is not None and ub is not None:
            self.patch_slip = np.clip(self.patch_slip, lb, ub)
        

        






    def __repr__(self):
        return f'Moment(Mw_mean={self.Mw_mean}, Mw_sigma={self.Mw_sigma}, shear_modulus={self.shear_modulus})'