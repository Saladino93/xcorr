"""
Utility to easily read data, sims results in one way.
"""

import numpy as np
import pathlib
import healpy as hp
import pymaster as nmt

class ReadSpectra(object):

    def __init__(self, filename:str = None, nside: int = 2048, lmax_binning: int = 4000, nlb: float = 50,
                 mc_correction = None, mask_correction = None):
        #self.spectra_path = pathlib.Path(spectra_path)
        lmaxpix = 3*nside-1
        pixwin = hp.pixwin(nside)[:lmax_binning+1]

        binning = nmt.NmtBin.from_lmax_linear(lmax_binning, nlb)

        self.ells = binning.get_effective_ells()
        if filename is not None:
            print("Loading workspace from file: {}".format(filename))
            workspace = self.load_workspace(str(filename))
            self.workspace = workspace
            M = self.workspace.get_bandpower_windows()[0, :, 0, :]
            self._pixwin = np.dot(M, pixwin)
            self.coupled_shape = (1, lmax_binning+1)

        self.mc_correction = mc_correction
        self.mask_correction = mask_correction

    @staticmethod
    def load_workspace(filename: str = ""):
        workspace = nmt.NmtWorkspace()
        workspace.read_from(filename)
        return workspace

    @staticmethod
    def _load(file:str):
        return np.loadtxt(file)

    def __call__(self, cls):

        cls_corrected = cls

        if self.mc_correction is not None:
            cls_corrected = cls*self.mc_correction

        if self.mask_correction is not None:
            cls_corrected = cls_corrected*self.mask_correction

        return cls_corrected

        """data = self._load(file).T
        
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
            
        return self.ells, self.cl, self.pixwin, self.pixwin_interp"""
        

    def couple_cell(self, cl_decoupled: np.ndarray):
        return self.workspace.couple_cell(cl_decoupled)
    
    def decouple_cell(self, cl_coupled: np.ndarray):
        return self.workspace.decouple_cell(cl_coupled)

    def bin_theory(self, cl_theory: np.ndarray):
        return self.decouple_cell(self.couple_cell([cl_theory]))[0]