"""
Shot noise utilities.
"""

import numpy as np

import healpy


def get_effective_n2(weights: np.ndarray, mask: np.ndarray, Omegapix: float) -> float:
    result = np.sum(weights)**2.
    result /= np.sum(weights**2.)
    result /= np.sum(mask)
    result /= Omegapix
    return result


def get_effective_n2_from_counts(workspace, coupled_shape, counts: np.ndarray, mask: np.ndarray, weights: np.ndarray = None) -> float:
    
    nside = healpy.npix2nside(len(mask))
    Omegapix = 4.*np.pi/len(mask)

    maskmean = np.mean(mask)

    if weights is None:
        weights = np.repeat(np.ones_like(counts), counts.astype(int), axis = -1)

    n2 = get_effective_n2(weights, mask, Omegapix)
    nshot = maskmean/n2

    nshot_effective = workspace.decouple_cell(nshot*np.ones(coupled_shape))[0]

    return nshot_effective

