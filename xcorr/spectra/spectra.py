import pymaster as nmt

import healpy as hp

import numpy as np


class AlphaNmtField(nmt.NmtField):
    """
    Convinience class to add an alpha multiplicative factor of the input maps.

    Useful when simulating rescaled gaussian delta_g maps and Poisson populating them.
    """
    def __init__(self, mask, maps, masked_on_input, alpha: float = 1., **kwargs):
        super().__init__(mask, maps, masked_on_input, **kwargs)
        self.alpha = alpha



class CrossCorrelate(object):
    """
    Class to compute the cross-correlation between two fields.

    Parameters
    ----------
    maskA : array_like
        Mask of the first field.
    maskB : array_like
        Mask of the second field.
    masked_on_input_A : bool
        Whether the mask is applied to the first field.
    masked_on_input_B : bool
        Whether the mask is applied to the second field.
    binning : pymaster.NmtBin
        Binning scheme to use.
    nside: int
        Nside of the maps.
    alpha: float
        Scaling factor for the input maps. This is for gaussian deltag simulations with an alpha factor to Poisson populate the maps.
    """

    def __init__(self, maskA, maskB, masked_on_input_A, masked_on_input_B, binning, nside: int):
        self.maskA = maskA
        self.maskB = maskB

        self.masked_on_input_A = masked_on_input_A
        self.masked_on_input_B = masked_on_input_B
        
        self.binning = binning
        self.ells = binning.get_effective_ells()

        fA = nmt.NmtField(maskA, [np.zeros_like(maskA)], masked_on_input = masked_on_input_A)
        fB = nmt.NmtField(maskB, [np.zeros_like(maskB)], masked_on_input = masked_on_input_B)

        workspace = nmt.NmtWorkspace()
        workspace.compute_coupling_matrix(fA, fB, binning)

        self.workspace = workspace

        pixwin = hp.pixwin(nside)
        M = workspace.get_bandpower_windows()[:, 0, :, 0]
        self._pixwin = np.dot(M, pixwin)

        self.pixwin_interp = np.interp(self.ells, np.arange(len(pixwin)), pixwin)

    def __call__(self, fA: AlphaNmtField, fB: AlphaNmtField = None):
        fB = fA if fB is None else fB
        factor = fA.alpha * fB.alpha
        cl_coupled = nmt.compute_coupled_cell(fA, fB)
        cl_decoupled = self.workspace.decouple_cell(cl_coupled)
        return cl_decoupled[0]/factor
    
    
    def couple_cell(self, cl_decoupled: np.ndarray):
        return self.workspace.couple_cell(cl_decoupled)
    
    def decouple_cell(self, cl_coupled: np.ndarray):
        return self.workspace.decouple_cell(cl_coupled)

    def bin_theory(self, cl_theory: np.ndarray):
        return self.decouple_cell(self.couple_cell([cl_theory]))

    @property
    def ell(self):
        return self.ells
    
    @property
    def pixwin(self):
        return self._pixwin



    


class MapsReader(CrossCorrelate):
    def __init__(self, maskA, maskB, masked_on_input_A, masked_on_input_B, binning, nside: int):
        super().__init__(maskA, maskB, masked_on_input_A, masked_on_input_B, binning, nside)

    def __call__(self, mappaA: np.ndarray, mappaB: np.ndarray = None, factorA: float = 1, factorB: float = 1):
        mappaB = mappaA if mappaB is None else mappaB
        fA = AlphaNmtField(self.maskA, [mappaA], masked_on_input = self.masked_on_input_A, alpha = factorA)
        fB = AlphaNmtField(self.maskB, [mappaB], masked_on_input = self.masked_on_input_B, alpha = factorB)
        return super().__call__(fA, fB)