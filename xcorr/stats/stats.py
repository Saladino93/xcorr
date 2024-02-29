import numpy as np
import scipy





def snr2(d: np.ndarray, invcov: np.ndarray, theory = 0.) -> float:
    chi2data = np.dot(invcov, d)
    chi2data = np.dot(d.T, chi2data)
    return chi2data

def get_pte_and_chi2_with_invcov(d: np.ndarray, invcov: np.ndarray):
    chi2data = snr2(d = d, invcov = invcov)
    dof = len(d)-1
    pte = 1 - scipy.stats.chi2.cdf(chi2data, dof)
    return pte, chi2data