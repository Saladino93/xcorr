import pymaster as nmt
import healpy as hp
import numpy as np


def get_w_factor(fieldA, fieldB):
    return np.mean(fieldA.get_mask() * fieldB.get_mask())

def gaussian_covariance_from_fields(fieldA, fieldB, fieldC, fieldD, wAB, wCD, cAC = None, cAD = None, cBC = None, cBD = None, spinA = 0, spinB = 0, spinC = 0, spinD = 0):
    cAC = cls_cov(fieldA, fieldC) if cAC is None else cAC
    cAD = cls_cov(fieldA, fieldD) if cAD is None else cAD
    cBC = cls_cov(fieldB, fieldC) if cBC is None else cBC
    cBD = cls_cov(fieldB, fieldD) if cBD is None else cBD
    return gaussian_covariance_from_spectra(fieldA, fieldB, fieldC, fieldD, wAB, wCD, cAC, cAD, cBC, cBD, spinA, spinB, spinC, spinD)


def gaussian_covariance_from_spectra(fieldA, fieldB, fieldC, fieldD, wAB, wCD, cAC, cAD, cBC, cBD, spinA = 0, spinB = 0, spinC = 0, spinD = 0):
    """
    Computes quick theory covariance matrix with Gaussian approximation.
    """

    cw = nmt.NmtCovarianceWorkspace()
    cw.compute_coupling_coefficients(fieldA, fieldB, fieldC, fieldD)
    cov = nmt.gaussian_covariance(cw,
                                spinA, spinB, spinC, spinD,
                                [cAC], [cAD], [cBC], [cBD],
                                wa = wAB, wb = wCD)
    return cov


def cls_cov(fieldA, fieldB):
    """
    Computes coupled cls for the covariance matrix.
    """
    return nmt.compute_coupled_cell(fieldA, fieldB)/get_w_factor(fieldA, fieldB)