"""
Shot noise utilities.
"""

import numpy as np

import healpy


def get_effective_n2(weights: np.ndarray, mask: np.ndarray, Omegapix: float, weights_sq: np.ndarray = None) -> float:
    result = np.sum(weights)**2.
    weights_sq = weights**2. if weights_sq is None else weights_sq
    result /= np.sum(weights_sq)
    result /= np.sum(mask)
    result /= Omegapix
    return result


def get_effective_n2_from_counts(workspace, coupled_shape, counts: np.ndarray, mask: np.ndarray, weights: np.ndarray = None, weights_sq: np.ndarray = None) -> float:
    """
    Get the effective n2 from the counts and the mask.

    For now we assume weights are all good to use. So, all inside the mask.
    """
    
    #nside = healpy.npix2nside(len(mask))
    Omegapix = 4.*np.pi/len(mask)

    maskmean = np.mean(mask)

    if weights is None:
        weights = np.repeat(np.ones_like(counts), counts.astype(int), axis = -1)

    n2 = get_effective_n2(weights, mask, Omegapix, weights_sq)
    nshot = maskmean/n2

    nshot_effective = workspace.decouple_cell(nshot*np.ones(coupled_shape))[0]

    return nshot_effective

