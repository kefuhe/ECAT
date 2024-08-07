"""
Theory: Conversion functions between the six fundamental material parameters in 3D Linear elasticity.
Example: https://en.wikipedia.org/wiki/Elastic_modulus
"""

import numpy as np


def get_constants_from_bulk_and_youngs(K, E):
    """
    Given Bulk (K) and Young's (E) moduli, what are the other elastic constants?

    :param K: bulk modulus
    :param E: young's modulus
    :returns: [lame1, G, poisson's ratio, and p-wave modulus]
    :rtype: list of 4 floats
    """
    lame1 = (3*K * (3*K - E)) / (9*K - E)
    G = (3*K*E) / (9*K - E)
    pr = (3*K - E) / (6*K)
    M = (3*K * (3*K + E) ) / (9*K - E)
    return [lame1, G, pr, M]


def get_constants_from_bulk_and_lame1(K, lame1):
    """
    Given Bulk modulus (K) and lame1 (lambda), what are the other elastic constants?

    :param K: bulk modulus
    :param lame1: lame's first parameter
    :returns: [E, G, poisson's ratio, and p-wave modulus]
    :rtype: list of 4 floats
    """
    E = (9*K * (K-lame1) ) / (3*K - lame1)
    G = (3 * (K-lame1) ) / 2
    pr = lame1 / (3*K - lame1)
    M = 3*K - 2*lame1
    return [E, G, pr, M]


def get_constants_from_bulk_and_shear(K, G):
    """
    Given Bulk modulus (K) and shear modulus (G), what are the other elastic constants?

    :param K: bulk modulus
    :param G: shear modulus
    :returns: [E, lame1, poisson's ratio, and p-wave modulus]
    :rtype: list of 4 floats
    """
    E = (9*K*G) / (3*K + G)
    lame1 = K - (2*G / 3)
    pr = (3*K - 2*G) / (2*(3*K+G))
    M = K + (4*G / 3)
    return [E, lame1, pr, M]


def get_constants_from_bulk_and_pr(K, nu):
    """
    Given Bulk modulus (K) and poisson's ratio (nu), what are the other elastic constants?

    :param K: bulk modulus
    :param nu: poisson's ratio
    :returns: [E, G, lame1, and p-wave modulus]
    :rtype: list of 4 floats
    """
    E = 3*K*(1-2*nu)
    lame1 = (3*K*nu)/(1+nu)
    G = (3*K*(1-2*nu))/(2*(1+nu))
    M = (3*K*(1-nu))/(1+nu)
    return [E, lame1, G, M]


def get_constants_from_bulk_and_pwave(K, M):
    """
    Given Bulk modulus (K) and P-wave modulus (M), what are the other elastic constants?

    :param K: bulk modulus
    :param M: p-wave modulus
    :returns: [E, G, lame1, and poisson's ratio]
    :rtype: list of 4 floats
    """
    E = (9*K*(M-K))/(3*K + M)
    lame1 = (3*K - M) / 2
    G = (3 * (M-K)) / 4
    pr = (3*K - M)/(3*K + M)
    return [E, lame1, G, pr]


def get_constants_from_youngs_and_lame1(E, lame1):
    """
    Given Young's modulus (E) and lame1 (lambda), what are the other elastic constants?

    :param E: Young's modulus
    :param lame1: lame's first parameter
    :returns: [K, G, poisson's ratio, and p-wave modulus]
    :rtype: list of 4 floats
    """
    R = np.sqrt(E*E + 9*lame1*lame1 + 2*E*lame1)
    K = (E + 3*lame1 + R) / 6
    G = (E - 3*lame1 + R) / 4
    pr = (2*lame1) / (E+lame1+R)
    M = (E-lame1+R) / 2
    return [K, G, pr, M]


def get_constants_from_youngs_and_shear(E, G):
    """
    Given Young's modulus (E) and shear modulus (G), what are the other elastic constants?

    :param E: Young's modulus
    :param G: shear modulus
    :returns: [K, lame1, poisson's ratio, and p-wave modulus]
    :rtype: list of 4 floats
    """
    K = (E*G) / (3*(3*G - E))
    lame1 = (G*(E-2*G)) / (3*G - E)
    pr = (E/(2*G)) - 1
    M = (G*(4*G - E)) / (3*G - E)
    return [K, lame1, pr, M]


def get_constants_from_youngs_and_nu(E, nu):
    """
    Given Young's modulus (E) and poisson's ratio (nu), what are the other elastic constants?

    :param E: Young's modulus
    :param nu: poisson's ratio
    :returns: [K, lame1, G, and p-wave modulus]
    :rtype: list of 4 floats
    """
    K = E / (3*(1-2*nu))
    lame1 = (E*nu) / ((1+nu) * (1-2*nu))
    G = E / (2*(1+nu))
    M = (E*(1-nu)) / ((1+nu)*(1-2*nu))
    return [K, lame1, G, M]


def get_constants_from_lame1_and_shear(lame1, G):
    """
    Given lame1 (lambda) and shear modulus (G), what are the other elastic constants?

    :param lame1: lame's first parameter
    :param G: shear modulus
    :returns: [K, E, poisson's ratio, and p-wave modulus]
    :rtype: list of 4 floats
    """
    K = lame1 + (2*G)/3
    E = (G * (3*lame1 + 2*G)) / (lame1 + G)
    pr = lame1 / (2 * (lame1 + G))
    M = lame1 + 2*G
    return [K, E, pr, M]


def get_constants_from_lame1_and_nu(lame1, nu):
    """
    Given lame1 (lambda) and poisson's ratio (nu), what are the other elastic constants?

    :param lame1: lame's first parameter
    :param nu: poisson's ratio
    :returns: [K, E, G, and p-wave modulus]
    :rtype: list of 4 floats
    """
    K = (lame1 * (1+nu)) / (3*nu)
    E = (lame1 * (1+nu) * (1-2*nu)) / nu
    G = (lame1 * (1-2*nu)) / (2*nu)
    M = (lame1 * (1-nu)) / nu
    return [K, E, G, M]


def get_constants_from_lame1_and_pwave(lame1, M):
    """
    Given lame1 (lambda) and P-wave modulus (M), what are the other elastic constants?

    :param lame1: lame's first parameter
    :param M: p-wave modulus
    :returns: [K, E, G, and poisson's ratio]
    :rtype: list of 4 floats
    """
    K = (M+2*lame1) / 3
    E = (M-lame1)*(M+2*lame1) / (M+lame1)
    G = (M-lame1) / 2
    pr = lame1 / (M+lame1)
    return [K, E, G, pr]


def get_constants_from_shear_and_nu(G, nu):
    """
    Given shear modulus (G) and poisson's ratio (nu), what are the other elastic constants?

    :param G: shear modulus
    :param nu: poisson's ratio
    :returns: [K, E, lame1, and p-wave modulus]
    :rtype: list of 4 floats
    """
    K = (2*G*(1+nu)) / (3 * (1-2*nu))
    E = 2*G*(1+nu)
    lame1 = (2*G*nu) / (1-2*nu)
    M = (2*G*(1-nu)) / (1-2*nu)
    return [K, E, lame1, M]


def get_constants_from_shear_and_pwave(G, M):
    """
    Given shear modulus (G) and p-wave modulus (M), what are the other elastic constants?

    :param G: shear modulus
    :param M: p-wave modulus
    :returns: [K, E, lame1, and pr]
    :rtype: list of 4 floats
    """
    K = M - (4*G)/3
    E = (G * (3*M - 4*G)) / (M-G)
    lame1 = M - 2*G
    pr = (M-2*G) / (2*M - 2*G)
    return [K, E, lame1, pr]


def get_constants_from_nu_and_pwave(nu, M):
    """
    Given shear modulus (G) and p-wave modulus (M), what are the other elastic constants?

    :param nu: poisson's ratio
    :param M: p-wave modulus
    :returns: [K, E, G, and lame1]
    :rtype: list of 4 floats
    """
    K = (M * (1+nu)) / (3 * (1-nu))
    E = (M * (1+nu) * (1-2*nu)) / (1-nu)
    lame1 = (M * nu) / (1-nu)
    G = (M * (1-2*nu) ) / (2 * (1-nu))
    return [K, E, G, lame1]
