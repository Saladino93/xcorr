import numpy as np

import healpy as hp

from typing import Tuple

import astropy.io as astro_io

import numba as nb


class Mask(object):
    """
    Mask object to handle masks and calculations with them
    """

    def __init__(self, mask: np.ndarray, threshold: float = 0.8):
        self.mask = mask
        self.bool_mask = mask > threshold
        self.masksum = self.sum(mask)
    
    @staticmethod
    #@nb.njit
    def _sum(x: np.ndarray, bool_mask: np.ndarray):
        """
        Makes a sum of the x array over the unmasked pixels
        """
        return np.sum(x[bool_mask])
    
    def sum(self, x: np.ndarray):
        """
        Makes a sum of the x array over the unmasked pixels
        """
        return self._sum(x, self.bool_mask)
    
    def mean(self, x: np.ndarray):
        """
        Makes a mean of the x array over the unmasked pixels, defined as

        $\sum x_i / \sum mask_i$

        """
        return self.sum(x)/self.masksum
    
    def divide(self, x: np.ndarray, y: np.ndarray):
        """
        Divides x array by y array, over the unmasked pixels
        """
        return self._divide(x, y, self.bool_mask)
    
    @staticmethod
    #@nb.njit
    def _divide(x: np.ndarray, y: np.ndarray, bool_mask: np.ndarray):
        """
        Divides x array by y array, over the unmasked pixels
        """
        out = np.zeros_like(x)
        out[bool_mask] = x[bool_mask]/y[bool_mask]
        return out

        


def read_catalog(filename: str):
    hdul = astro_io.fits.open(name = filename)
    data = hdul[1].data
    return data

def save_catalog(filename: str, columnsnames: list, columns: list, overwrite: bool = True):
    '''
    https://het.as.utexas.edu/HET/Software/Astropy-1.0/io/fits/index.html#creating-a-new-table-file
    '''

    cols = [astro_io.fits.Column(name, format = 'E', array = arr) for name, arr in zip(columnsnames, columns)]
    cols = astro_io.fits.ColDefs(cols)
    tbhdu = astro_io.fits.BinTableHDU.from_columns(cols)
    return tbhdu.writeto(filename, overwrite = overwrite)

def get_nz(redshifts: np.ndarray, weights: np.ndarray = None, nbins: int = 20) -> Tuple[np.ndarray, np.ndarray]:

    if type(nbins) == int:
        bins = nbins
    else:
        bins = nbins
    #CREATE dn/dz
    #histog = np.histogram(redshifts, galaxyzbins, weights = weights)
    histog = np.histogram(redshifts, weights = weights, bins = bins, density = None)
    z, nz = histog[1], histog[0]
    z = (z[1:]+z[:-1])/2.
    return z, nz



#@numba.jit(nopython = True)
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

def get_nz_from_catalog(catalog_keys: dict, filename: str, zmin: float, zmax: float, nbins: int = 20, redshiftkey: str = 'redshifts', windowkey: str = 'windows', weightskey : str = 'weights') -> Tuple[np.ndarray, np.ndarray]:

    catalog = read_catalog(filename)
    all_values_of_reference = catalog[catalog_keys[redshiftkey]]
    selector = get_selector_from_array_values(bottom = zmin, top = zmax, all_values = all_values_of_reference)

    all_values_of_reference = catalog[catalog_keys[windowkey]]
    try:
        weights = catalog[catalog_keys[weightskey]]
    except:
        print("Setting unit weights!!")
        weights = np.ones_like(all_values_of_reference)

    data = np.c_[all_values_of_reference, weights]

    data = selector(data)

    redshifts, weights = data[:, 0], data[:, 1]
    
    return get_nz(redshifts = redshifts, weights = weights, nbins = nbins)


def get_redshift_bins_list_from_edges(redshift_bin_edges: np.ndarray, entire_range: bool = True):
    up, down = redshift_bin_edges[1:], redshift_bin_edges[:-1]
    extra = [[redshift_bin_edges[0], redshift_bin_edges[-1]]]if entire_range else []
    return [[low, high] for low, high in zip(down, up)]+extra

def get_binary_mask_from_dowgrading_completeness_from_data(nside: int, fracgood: np.ndarray, healpix_positions: np.ndarray, 
                                                          nside_out: int, 
                                                          threshold: float = 0.) -> np.ndarray:

    completeness_map = get_completeness_map(nside, fracgood, healpix_positions)
    return get_binary_mask_from_downgrading_completeness(nside_out = nside_out, completeness_map = completeness_map, threshold = threshold)


def get_completeness_map(nside: int, fracgood: np.ndarray, healpix_positions: np.ndarray, fill_value = hp.pixelfunc.UNSEEN) -> np.ndarray:
    shape = hp.nside2npix(nside)
    completeness_map = np.ones(shape)*fill_value
    completeness_map[healpix_positions] = fracgood
    return completeness_map


def get_binary_mask_from_completeness(completeness_map: np.ndarray, threshold: float = 0.8) -> np.ndarray:
    sel = completeness_map > threshold
    mask = np.zeros_like(completeness_map)
    mask[sel] = 1. #mask = 1*sel
    return mask

def downgrade_map(input_map: np.ndarray, nside_out: int) -> np.ndarray:
    return hp.pixelfunc.ud_grade(input_map, nside_out = nside_out)

def get_binary_mask_from_downgrading_completeness(nside_out: int, completeness_map: np.ndarray, threshold: float = 0.):
    completeness_map_downgraded = downgrade_map(completeness_map, nside_out = nside_out)
    return get_binary_mask_from_completeness(completeness_map_downgraded, threshold = threshold)


#numba version?
def get_pixels_from_subsamplesnumber(pixels: np.ndarray, weights:np.ndarray, subsamples_number: int):

    if subsamples_number == 1:
        pixels = [pixels]
        weights = [weights]
    else:
        #seed??
        rng = np.random.default_rng()
        data = np.vstack((pixels, weights))#2 rows x ncols
        rng.shuffle(data, axis = 1)
        splitted_data = np.array_split(data, subsamples_number, axis = 1)
        pixels = [data[0, :].astype(int) for data in splitted_data]
        weights = [data[1, :] for data in splitted_data]

    return pixels, weights


def make_map_from_values_at_decs_ras(nside: int, decs: np.ndarray, ras: np.ndarray, values: np.ndarray) -> np.ndarray:
    '''
    Given coordinates in the form of decs, ras (in degrees), and values, 
    returns a healpix map with some nside resolution.

    Parameters
    ----------
    Returns
    ----------
    Map counts
    '''
    
    shape = hp.nside2npix(nside)
    pixels = hp.ang2pix(nside, ras, decs, lonlat = True)
    map = np.zeros(shape)
    map[pixels] = values

    return map

#numba version
@nb.njit
def add_at(shape, pixs, wghts):
    mappa = np.zeros(shape)
    for i in nb.prange(shape):
        mappa[pixs[i]] += wghts[i]
    return mappa

def get_weighted_counts_from_coords(nside: int, decs: np.ndarray, ras: np.ndarray, weights: np.ndarray, subsamples_number: int = 1) -> np.ndarray:
    '''
    Given coordinates in the form of decs, ras (in degrees), and weights, 
    returns a healpix map with some nside resolution.

    Parameters
    ----------
    Returns
    ----------
    Map counts
    '''
    
    shape = hp.nside2npix(nside)
    pixels = hp.ang2pix(nside, ras, decs, lonlat = True) #all the pixels of the catalog

    #pixels = hp.ang2pix(nside, (90.0-decs)/180.0*np.pi, ras/180.0*np.pi)

    pixels, weights = get_pixels_from_subsamplesnumber(pixels, weights, subsamples_number)

    make_histogram = lambda pixs, wghts: np.histogram(pixs, bins = shape, weights = wghts, range = [0, shape], density = False)[0].astype(np.float32)

    def make_histogram(pixs, wghts):
        mappa = np.zeros(shape, dtype = float)        
        np.add.at(mappa, pixs, wghts)
        #mappa = add_at(shape, pixs, wghts)
        return mappa

    counts = [make_histogram(pixs, wghts) for pixs, wghts in zip(pixels, weights)]
    counts = counts[0] if subsamples_number == 1 else counts

    return counts


def get_average_number_density_gg_per_arcmin2(counts: np.ndarray, mask: np.ndarray, pix_area_out_arcmin2: float):
    '''
    Returns average number density of galaxies in arcmin^{-2} 


    Parameters
    -----------
    Returns
    -----------

    '''

    average_number_density_gg = np.nansum(counts)/(pix_area_out_arcmin2*np.sum(mask))
    return average_number_density_gg


def get_average_number_density_gg(counts: np.ndarray, mask: np.ndarray, completeness_map: np.ndarray, pix_area_out_arcmin2: float):
    '''
    Returns average number density of galaxies in arcmin^{-2} 

    \bar{n} = \frac{\sum_i N_i}{\sum_i f_i \Omega}, \sum_i is over unmasked pixels, \Omega pixel area, for healpix is constant for all pixels

    See Formula (19) of Camacho et al. 2019

    Parameters
    -----------
    Returns
    -----------

    '''
    #return np.nansum(counts/completeness_map*mask)/(pix_area_out_arcmin2*np.nansum(completeness_map*mask))
    return np.nansum(counts*mask)/(pix_area_out_arcmin2*np.nansum(completeness_map*mask))


def get_effective_counts(counts: np.ndarray, completeness_map: np.ndarray) -> np.ndarray:
    return counts/completeness_map

def get_number_density_map_gg(counts: np.ndarray, completeness_map: np.ndarray, pix_area_out_arcmin2: float) -> np.ndarray:
        
    number_density_gg_map = get_effective_counts(counts, completeness_map)/pix_area_out_arcmin2
    
    return number_density_gg_map


def get_density_contrast_map_gg(counts: np.ndarray, mask: np.ndarray, completeness_map: np.ndarray, pix_area_out_arcmin2: float) -> np.ndarray:

    average_number_density_gg = get_average_number_density_gg(counts = counts, mask = mask, completeness_map = completeness_map, pix_area_out_arcmin2 = pix_area_out_arcmin2)

    number_density_gg_map = get_number_density_map_gg(counts = counts, completeness_map = completeness_map, pix_area_out_arcmin2 = pix_area_out_arcmin2)
    
    over_density_gg_map = np.nan_to_num(number_density_gg_map)/average_number_density_gg-1
    
    return over_density_gg_map