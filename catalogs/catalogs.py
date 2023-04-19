from typing import Dict

import numpy as np

import catalogs_utils

import utils

import numba

#from numba import jit


class DataCatalogMapper(object):

    '''
    '''
    
    def __init__(self, data: Dict):
        '''
        '''

        self.ras_ = ''
        self.decs_ = ''
        self.weights_ = ''
        self.redshifts_ = ''
        self.redshifts_uncertainties_ = ''

        self._weights_values = None

        self.data = data

    def get_data_from_dict_key(self, key: str, dict_: Dict) -> np.ndarray:
        return dict_[key]

    def get_data_from_key(self, key: str) -> np.ndarray:
        return self.get_data_from_dict_key(key = key, dict_ = self.data)

    #def set_data_from_key(self, newvalue: np.ndarray, key: str) -> np.ndarray:
    #    self.data[key] = newvalue
    #    return newvalue

    @property    
    def ras(self):
        return self.get_data_from_key(self.ras_)

    @ras.setter
    def ras(self, value: str):
        self.ras_ = value

    @property    
    def decs(self):
        return self.get_data_from_key(self.decs_)

    @decs.setter
    def decs(self, value: str):
        self.decs_ = value

    @property    
    def weights(self):
        if self._weights_values is None:
            self._weights_values = self.get_data_from_key(self.weights_)
        return self._weights_values

    @weights.setter
    def weights(self, value: str):
        self.weights_ = value

    @property    
    def redshifts(self):
        return self.get_data_from_key(self.redshifts_)

    @redshifts.setter
    def redshifts(self, value: str):
        self.redshifts_ = value

    @property    
    def redshifts_uncertainties(self):
        return self.get_data_from_key(self.redshifts_uncertainties_)

    @redshifts_uncertainties.setter
    def redshifts_uncertainties(self, value: str):
        self.redshifts_uncertainties_ = value



class Catalog(DataCatalogMapper):

    '''
    Parameters
    ----------
    Returns
    ----------
    '''

    def __init__(self, data: Dict):
        '''
        Parameters
        ----------
        Returns
        ----------
        '''
        super().__init__(data = data)


    def get_map_counts_at_redshifts(self, nside: int = 512, bottom_z: float = None, top_z: float = None, subsamples_number: int = 1):
        '''

        When some redshift is none, assume maximum or minimum redshift of the whole catalog.

        Parameters
        ----------
        Returns
        ----------
        '''
        if bottom_z is None:
            bottom_z = self.redshifts.min()
        if top_z is None:
            top_z = self.redshifts.max()

        selector = utils.get_selector_from_array_values(bottom = bottom_z, top = top_z, all_values = self.redshifts)
        decs = selector(self.decs)
        ras = selector(self.ras)
        weights = selector(self.weights)
        return catalogs_utils.get_weighted_counts_from_coords(nside = nside, decs = decs, ras = ras,
                                                    weights = weights, subsamples_number = subsamples_number)
        
    @staticmethod
    def select(all_values: np.ndarray, bottom: float, top: float, all_values_of_reference: np.ndarray = None) -> np.ndarray:
        if all_values_of_reference is None:
            all_values_of_reference = all_values
            
        assert len(all_values) == len(all_values_of_reference)
        selector = utils.get_selector_from_array_values(bottom = bottom, top = top, all_values = all_values_of_reference)
        return selector(all_values)

    def select_redshifts(self, bottom_z: float, top_z: float) -> np.ndarray:
        return self.select(all_values = self.redshifts, bottom = bottom_z, top = top_z, all_values_of_reference = self.redshifts)
