"""
Utility to easily read data, sims results in one way.
"""

import numpy as np
import pathlib
import healpy as hp


class ReadSpectra(object):

    def __init__(self, mc_correction: callable = None, mask_correction: callable = None, include_shot: bool = False, filename:str = None, nside: int = 2048, lmax_binning: int = 4000):
        #self.spectra_path = pathlib.Path(spectra_path)
        self.include_shot = include_shot
        self.mc_correction = mc_correction
        self.mask_correction = mask_correction
        pixwin = hp.pixwin(nside)[:lmaxpix+1]

        self.filename = filename
        if filename is not None:
            print("Loading workspace from file: {}".format(filename))
            workspace = self.load_workspace(str(filename))
            self.workspace = workspace
            M = self.workspace.get_bandpower_windows()[0, :, 0, :]
            self._pixwin = np.dot(M, pixwin)
            self.coupled_shape = (1, lmax_binning+1)

    @staticmethod
    def _load(file:str):
        return np.loadtxt(file)

    def __call__(self, file:str):

        data = self._load(file).T
        
        if self.include_shot:
            ells, cl, shot, pixwin, pixwin_interp = data
            cl -= shot
        else:
            ells, cl, pixwin, pixwin_interp = data

        self.ells = ells.astype(int)
        self.pixwin = pixwin
        self.pixwin_interp = pixwin_interp
        self.cl = cl
        self.cl *= self.mc_correction(ells) if self.mc_correction is not None else 1.
        self.cl *= self.mask_correction(ells) if self.mask_correction is not None else 1.
            
        return self.ells, self.cl, self.pixwin, self.pixwin_interp
        

    def couple_cell(self, cl_decoupled: np.ndarray):
        return self.workspace.couple_cell(cl_decoupled)
    
    def decouple_cell(self, cl_coupled: np.ndarray):
        return self.workspace.decouple_cell(cl_coupled)

    def bin_theory(self, cl_theory: np.ndarray):
        return self.decouple_cell(self.couple_cell([cl_theory]))[0]