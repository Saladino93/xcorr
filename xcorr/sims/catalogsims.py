import numpy as np

import healpy as hp

from xcorr.simstools import catutils

import pathlib

import logging

log = logging.getLogger(__name__)

class SimCatalog(object):
    
    def __init__(self, nside: int, directory: pathlib.Path):
        self.nside = nside
        self.directory = directory


    @staticmethod
    def check_if_binary_mask(mask: np.ndarray):
        if np.all(np.isin(mask, [0, 1])):
            return True
        else:
            return False

    def _delta_g_to_number_counts(self, No_observed, delta_g_input_real, seed, mask, weights, alpha, alpha_delta: float = 1., cross_check_negative: bool = True):
        '''
        This function transforms a delta_g map to a number counts map. For alpha < 1 see https://arxiv.org/pdf/1708.01536.pdf

        Parameters
        ----------
        No_observed : int
            Number of observed galaxies.
        delta_g_input_real : np.ndarray
            Delta_g map.
        seed : int
            Seed for the random number generator.
        mask : np.ndarray
            Mask of the map. e.g. completeness mask.
        weights : np.ndarray
            Weights of the map.
        alpha : float
            Scaling factor for the delta_g map.

        Returns
        -------
        np.ndarray
            Number counts map.
        '''

        #this is here so that if I do not want to sample, e.g. because I have a kappa map, I can just return the input map
        if No_observed == 0:
            return delta_g_input_real

        mappa = (delta_g_input_real*alpha_delta+1.)

        if cross_check_negative:
            selection = mappa < 0
            if np.any(selection):
                log.warning(f"Negative delta_g detected. Setting to 0. Seed: {seed}, len(selection): {len(selection[selection])}")
                mappa[selection] = 0.


        
        Npix = np.sum(mask) #len(mappa[mask != 0])
        Ngal = mappa*No_observed/Npix
        #Ngal = hp.alm2map(hp.map2alm(Ngal, pol = False, use_pixel_weights = True), nside, pixwin = True)#NOTE: PIXWIN is already applied on the theory power spectra
        #assume mask map is "inside" the weights map 
        if type(weights) is np.ndarray:
            inverse_weights = weights.copy()
            inverse_weights[weights != 0] = 1./weights[weights != 0]
        else:
            inverse_weights = 1./weights

        Ngal = np.nan_to_num(Ngal)*mask*inverse_weights/alpha**2.
        
        if alpha == 1:
            Ngal[np.where(Ngal < 0.)] = 0. #setting to -1 the delta_g_input_real
            
        return (self.poisson_sampling(Ngal, seed)).astype(int)

    @staticmethod
    def poisson_sampling(counts_map: int, seed: int):
        rng = np.random.default_rng(seed = seed)
        return rng.poisson(lam = counts_map)
    
    
    def create_field_with_systematics(self, Ngal: np.ndarray, weight_map: np.ndarray, seed: int) -> np.ndarray:
        Ngal_corrected = Ngal/weight_map
        return self.poisson_sampling(Ngal_corrected, seed)

    
    def delta_g_to_number_counts(self, seed, No_observed, delta_g_input_real, mask, weights, alpha, alpha_delta: float = 1.):
        if type(No_observed) is not list:
            No_observed = [No_observed]
            delta_g_input_real = [delta_g_input_real]
            weights = [weights]

        return [self._delta_g_to_number_counts(No, delta_g, seed, mask, weight, alpha, alpha_delta) for No, delta_g, weight in zip(No_observed, delta_g_input_real, weights)]
    

    def save_catalog(self, filename: str, seed: int, No_observed: int, delta_g_input_real: np.ndarray, mask: np.ndarray, weight_maps: np.ndarray, alpha: float, z: np.ndarray, nz: np.ndarray, alpha_delta: float = 1., columnsnames: list = ['RA', 'DEC', 'Z', 'ZBIN', 'WEIGHT']):
        number_counts = self.delta_g_to_number_counts(seed, No_observed, delta_g_input_real, mask, weight_maps, alpha, alpha_delta)
        Npix = len(mask)
        catutils.save_catalog(filename = self.directory/f'{filename}.fits', columnsnames = columnsnames, columns = self._get_catalog(seed, Npix, self.nside, number_counts, z, nz, weight_maps))
        return number_counts
    
    @staticmethod
    def inverse(array: np.ndarray):
        print(array)
        array[array != 0] = 1./array[array != 0]
        print(array)
        return array

    def _get_catalog(self, seed: int, Npix: int, nside: int, number_counts: np.ndarray, z: np.ndarray, nz: np.ndarray, weights_maps: np.ndarray):

        if type(number_counts) is not list:
            number_counts = [number_counts]
            z = [z]
            nz = [nz]

        selections = [number_ != 0 for number_ in number_counts]

        ipix = np.arange(0, Npix)
        ras, decs = hp.pixelfunc.pix2ang(nside, ipix, lonlat = True)
        coords = np.vstack((ras, decs))

        #sample angular coordinates
        catalog = np.hstack([np.repeat(coords, number_counts_.astype(int), axis = -1) for number_counts_ in number_counts])

        log.info("Sampling redshifts")
        print("Sampling redshifts")
        #sample redshifts
        zs = np.hstack([catutils.sample_from_nz_pdf(z[ii], nz[ii], np.sum(number_counts_), seed = seed, zconst = ii) for ii, number_counts_ in enumerate(number_counts)])
        print("Done")

        ras = catalog[0, :]
        decs = catalog[1, :]

        del catalog


        """
        if np.all(weights_maps == ones):
            fake_weights = np.ones(catalog.shape[-1])
            #weights_maps = []
            #for jj in range(len(zkeys)):
            #    weights_maps += [fake_weights]
            #fake_weights = np.concatenate(weights_maps)
        else:
        """

        #weights_maps = [np.ones(Npix) for _ in range(len(number_counts))]
        #*self.inverse(number_counts_.astype(int))

        fake_weights = np.concatenate([np.repeat(weights[selection]/number_counts_.astype(float)[selection], number_counts_.astype(int)[selection], axis = -1) for selection, weights, number_counts_ in zip(selections, weights_maps, number_counts)])

        #weights_sq = np.concatenate([np.repeat(weights[selection]**2/number_counts_.astype(float)[selection], number_counts_.astype(int)[selection], axis = -1) for selection, weights, number_counts_ in zip(selections, weights_maps, number_counts)])
        
        #print(fake_weights.shape, fake_weights)

        weights = fake_weights

        #just a label to identify each redshift bin
        zbin = np.concatenate([i*np.ones(np.sum(elements[1][elements[0]])) for i, elements in enumerate(zip(selections, number_counts))])

        return (ras, decs, zs, zbin, weights)