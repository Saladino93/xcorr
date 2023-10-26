import healpy as hp

import pymaster as nmt

import numpy as np

#from numba import jit

from typing import List, Callable, Dict

import pickle

import pathlib

import scipy

#import sympy as sp


def bin_theory(l, cl, bin_edges):
    sums = scipy.stats.binned_statistic(l, l, statistic="sum", bins=bin_edges)
    cl = scipy.stats.binned_statistic(l, l*cl, statistic="sum", bins=bin_edges)
    cl = cl[0] / sums[0]
    bin_centers = (bin_edges[1:] + bin_edges[:-1])/2
    return bin_centers, cl


def get_general_propagated_error(first_derivatives: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    '''
    Valid as long as you have first-order Taylor expansion.

    Parameters
    ----------
    first_derivatives: np.ndarray
        first derivatives vector of shape (n_vars, n_bins)
    covariance: np.ndarray
        covariance matrix for the observables of shape (n_vars, n_vars, n_bins)

    Returns
    -------
    error: np.ndarray
        error per bin of shape (n_bins)
    '''
    error = np.einsum('il, ijl, jl -> l ', first_derivatives, covariance, first_derivatives)
    
    return np.sqrt(error)



def build_cov_from_dict(ells: np.ndarray, delta_ells: np.ndarray,dictionary: Dict, relevant_keys: List[str], window_key: str):
    '''
    dictionary has to contain all the possible cross-correlations in relavant_keys, as well the masks for each field

    '''

    def get_from_dict(dictionary_, A, B):
        try:
            key_ = A+B
            result = dictionary_[key_]
        except:
            key_ = B+A
            result = dictionary_[key_]
        return result

    N_fields = len(relevant_keys)
    N_bins = len(ells)
    
    #result = np.zeros((N_fields, N_fields, N_bins))
    result = {}
    key_combs = list(itertools.combinations_with_replacement(relevant_keys, 2))

    #kk, kg, ks, gg, gs, ss
    
    for i, comb_1 in enumerate(key_combs):
        X, Y = comb_1
        for j, comb_2 in enumerate(key_combs):
            W, Z = comb_2
            XW = get_from_dict(dictionary, X, W)
            YZ = get_from_dict(dictionary, Y, Z)
            XZ = get_from_dict(dictionary, X, Z)
            YW = get_from_dict(dictionary, Y, W)
            wX = get_from_dict(dictionary, X, window_key)
            wY = get_from_dict(dictionary, Y, window_key)
            wW = get_from_dict(dictionary, W, window_key)
            wZ = get_from_dict(dictionary, Z, window_key)
            
            common_window_XY = wX*wY
            common_window_WZ = wW*wZ
            common_window = common_window_XY*common_window_WZ

            #<w_1Xw_2Yw_3Ww_4Z> -> <w_1_w2_w_3w_4>/<w_1w_2>/<w_3w_4>
            wfactor = np.nanmean(common_window)/np.nanmean(common_window_XY)/np.nanmean(common_window_WZ)
            fsky = np.nansum(common_window)/len(common_window)

            #result[i%N_fields, j%N_fields, :] = get_cov_between_XY_WZ(ells, delta_ells, fsky, wfactor, XW, YZ, XZ, YW)
            #result[j%N_fields, i%N_fields, :] = result[i%N_fields, j%N_fields, :]
            result[f'{X}-{Y}', f'{W}-{Z}'] = get_cov_between_XY_WZ(ells, delta_ells, fsky, wfactor, XW, YZ, XZ, YW)
    return result

def get_cov_between_XY_WZ(ells, delta_ells, fsky, wfactor, XW, YZ, XZ, YW):
    Nmodes = (2*ells+1)*delta_ells*fsky/wfactor #wfactor multiplies spectra, so here I divide
    return _get_cov_between_XY_WZ_with_Nmodes(Nmodes, XW, YZ, XZ, YW)

def _get_cov_between_XY_WZ_with_Nmodes(Nmodes, XW, YZ, XZ, YW):
    return (XW*YZ+XZ*YW)/Nmodes

def snr2(d: np.ndarray, invcov: np.ndarray, theory = 0.) -> float:
    chi2data = np.dot(invcov, d)
    chi2data = np.dot(d.T, chi2data)
    return chi2data

def get_pte_and_chi2_with_invcov(d: np.ndarray, invcov: np.ndarray):
    chi2data = snr2(d = d, invcov = invcov)
    dof = len(d)-1
    pte = 1 - scipy.stats.chi2.cdf(chi2data, dof)
    return pte, chi2data

def save_dicts(names: List[str], dictionaries: List[Dict]):
    for name, dictionary in zip(names, dictionaries):
        with open(name, 'wb') as f:
            pickle.dump(dictionary, f)

def read_dicts(names: List[str]) -> List[Dict]:

    def read_dict(name: str):
        with open(name, 'rb') as f:
            dictionary = pickle.load(f)
        return dictionary

    dictionaries = list(map(read_dict, names))
    return dictionaries

def save_maps(names: List[str], maps: List[np.ndarray], saving_function: Callable = np.save, common_dir: str = '', **kwargs):
    """
    Saves a list of maps, with their filenames.

    Parameters
    ----------
    names: List[str]
        List of output filenames
    maps: List[np.ndarray]
        List of maps
    saving_function: Callable (default np.save)
        Saving function. Has to take as first argument the name, second argument the object
    commondir: str (default '')

    Returns
    -------
    None
    """

    save = lambda name, mappa: saving_function(name, mappa, **kwargs)
    [save(str(common_dir/pathlib.Path(name)), mappa) for name, mappa in zip(names, maps)]

    return None

def read_maps(names: List[str], reading_function: Callable = np.load, ext: str = '.npy', common_dir: str = '', **kwargs) -> List[np.ndarray]:
    """
    Read a list of maps from their filenames.

    Parameters
    ----------
    names: List[str]
        List of output filenames
    reading_function: Callable (default np.save)
        Reading function. Has to take as first argument the name
    ext: str (default '.npy')
        Extension of the file
    Returns
    -------
    mappe: List[np.ndarray]
        The output list of maps
    """

    read = lambda name: reading_function(name, **kwargs)
    mappe = [read(str(common_dir/pathlib.Path(name))+ext) for name in names]

    return mappe



def apodize_mask(mask: np.ndarray, aposize: float = 0.1, apotype: str = 'C1') -> np.ndarray:

    return nmt.mask_apodization(mask, aposize = aposize, apotype = apotype)


def downgrade_map(input_map: np.ndarray, nside_out: int) -> np.ndarray:
    return hp.pixelfunc.ud_grade(input_map, nside_out = nside_out)


def get_selector_from_array_values(bottom: float, top: float, all_values: np.ndarray):
    '''
    Given a list of values in an array, gives selector for which value is in [bottom, top] 

    Parameters
    ----------
    Returns
    ----------
    '''
    selection = (all_values >= bottom) & (all_values <= top)
    selector = lambda x: x[selection]
    return selector


def select_from_array_values(bottom: float, top: float, all_values: np.ndarray):
    '''
    Given a list of values in an array, select values in the interval [bottom, top] 

    Parameters
    ----------
    Returns
    ----------
    '''

    selector = get_selection_from_array_values(bottom = bottom, top = top, all_values = all_values)
    return selector(all_values)

def list_alm_copy(alm_list: np.ndarray or list, mmaxin:int or None, lmaxout:int, mmaxout:int):
    if type(alm_list) == list:
        return [alm_copy(a, mmaxin, lmaxout, mmaxout) for a in alm_list]
    else:
        return alm_copy(alm_list, mmaxin, lmaxout, mmaxout)
#copied from delensalot
def alm_copy(alm:np.ndarray, mmaxin:int or None, lmaxout:int, mmaxout:int):
    """Copies the healpy alm array, with the option to change its lmax

        Parameters
        ----------
        alm :ndarray
            healpy alm arrays to copy.
        mmaxin: int or None
            mmax parameter of input array (can be set to None or negative for default)
        lmaxout : int
            new alm lmax
        mmaxout: int
            new alm mmax


    """
    alms = np.atleast_2d(alm)
    ret = []
    for alm in alms:
        lmaxin = hp.Alm.getlmax(alm.size, mmaxin)
        if mmaxin is None or mmaxin < 0: mmaxin = lmaxin
        if (lmaxin == lmaxout) and (mmaxin == mmaxout):
            ret.append(np.copy(alm))
        else:
            _ret = np.zeros(hp.Alm.getsize(lmaxout, mmaxout), dtype=alm.dtype)
            lmax_min = min(lmaxout, lmaxin)
            for m in range(0, min(mmaxout, mmaxin) + 1):
                idx_in =  m * (2 * lmaxin + 1 - m) // 2 + m
                idx_out = m * (2 * lmaxout+ 1 - m) // 2 + m
                _ret[idx_out: idx_out + lmax_min + 1 - m] = alm[idx_in: idx_in + lmax_min + 1 - m]
            ret.append(_ret)
    ret = np.array(ret)
    if ret.shape[0] == 1:
        return ret[0]
    else:
        return ret
