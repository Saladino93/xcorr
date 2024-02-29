"""
Generates conditioned Gaussian simulations from a given map, and power spectra.
"""

import numpy as np

import healpy as hp

from typing import Callable

import utils



class ConditionedSims(object):
    """
    Class to generate conditioned Gaussian simulations from a given map, and power spectra.
    """

    def __init__(self, Nfields: int, get_AB: Callable, realized_field_index: int = 0):
        """
        Parameters
        ----------
        Nfields : int
            Number of fields.
        get_AB : callable
            Function that returns the cross-correlation power spectrum between field A and B.
        realized_field_index : int
            Index of the fixed field in the power spectra function.
        """

        self.Nfields = Nfields
        self.get_AB = get_AB
        indices = utils.get_indices_for_cls_list(Nfields)


    def generate(self):
        """
        Generates the conditioned Gaussian simulations.
        """

        return generate_gaussian_conditioned_sims(self.input_kappa_alm, self.input_theory_dir, self.nside_map, self.seed, self.Nfields)





def procesess_cls_list(Nfields: int, get_gg: callable, get_kg: callable, get_kk: callable):

    indices = get_indices_for_cls_list(Nfields)

    length = min(len(get_gg(0, 0)), len(get_kg(0)), len(get_kk()))
        
    clsfields = [get_gg(i, j) for i, j in zip(indices[0], indices[1])]
    clsfieldscross = [get_kg(i) for i in range(Nfields)]
    
    clsfieldscross += [get_kk()[:length]]

    cls = clsfields + clsfieldscross


    #now, make some processing so that I have ready info for sims generation
    clsfieldscross = np.array(clsfieldscross)
    clsfields = np.array(clsfields)

    kk = clsfieldscross[-1]
    
    clsfieldscross = clsfieldscross[:-1]
    filter_correlated = clsfieldscross/kk #this is a numpy array, shape of (Nfield+1, Nls)

    clsfieldscross = clsfieldscross/np.sqrt(kk) #remove the last one, which is PxP

    matrix = np.einsum('i..., j...', clsfieldscross, clsfieldscross)
    matrix = np.array([matrix[:, i, j] for i, j in zip(indices[0], indices[1])])
    covij = np.array(clsfields)
    uncorrelated = covij-matrix 

    return filter_correlated, uncorrelated




def generate_gaussian_conditioned_sims(input_alm: np.ndarray, seed: int)
                                       
                                       
                                       
                                       
, input_theory_dir: str, nside_map: int, seed: int, Nfields: int):
    """
    """

    get_kk, get_kg, get_gg = prepare_get_functions_for_conditioned_sims(input_theory_dir)

    lmax = 3*nside_map-1
    input_kappa_alm = _alm_copy(input_kappa_alm, lmax, lmax)

    rng = np.random.default_rng(seed = seed)
    
    filter_correlated, filters_uncorrelated = procesess_cls_list(Nfields, get_gg, get_kg, get_kk)

    total_g = get_total_g(input_kappa_alm, filter_correlated, filters_uncorrelated, seed)

    gmaps = get_total_g_real(total_g_alm = total_g, nside = nside_map)
    
    return 0, gmaps