import pymaster as nmt

import healpy as hp

import numpy as np

from xcorr.spectra import shotutils


class AlphaNmtField(nmt.NmtField):
    """
    Convinience class to add an alpha multiplicative factor of the input maps.

    Useful when simulating rescaled gaussian delta_g maps and Poisson populating them.
    """
    def __init__(self, mask, maps, masked_on_input, alpha: float = 1., **kwargs):
        super().__init__(mask = mask, maps = maps, masked_on_input = masked_on_input, **kwargs)
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

    def __init__(self, maskA, maskB, masked_on_input_A, masked_on_input_B, binning, nside: int, filename: str = None, lmax_sht: int = -1):
        self.maskA = maskA
        self.maskB = maskB

        self.masked_on_input_A = masked_on_input_A
        self.masked_on_input_B = masked_on_input_B

        self.nside = nside
        
        self.binning = binning
        self.ells = binning.get_effective_ells()

        fA = nmt.NmtField(maskA, [np.zeros_like(maskA)], masked_on_input = masked_on_input_A, lmax_sht = lmax_sht)
        fB = nmt.NmtField(maskB, [np.zeros_like(maskB)], masked_on_input = masked_on_input_B, lmax_sht = lmax_sht)

        self.filename = filename
        if filename is not None:
            self.load_workspace(filename)
        else:
            workspace = nmt.NmtWorkspace()
            workspace.compute_coupling_matrix(fA, fB, binning)

        self.workspace = workspace

        lmaxpix = min(3 * nside - 1, lmax_sht)
        pixwin = hp.pixwin(nside)[:lmaxpix+1]
        M = workspace.get_bandpower_windows()[0, :, 0, :]
        self._pixwin = np.dot(M, pixwin)

        self.pixwin_interp = np.interp(self.ells, np.arange(len(pixwin)), pixwin)

        self.coupled_shape = nmt.compute_coupled_cell(fA, fB).shape



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
        return self.decouple_cell(self.couple_cell([cl_theory]))[0]

    @property
    def ell(self):
        return self.ells
    
    @property
    def pixwin(self):
        return self._pixwin
    
    def save_workspace(self, filename: str = None):
        filename = self.filename if filename is None else filename
        self.workspace.write_to(filename)

    def load_workspace(self, filename: str = None):
        self.workspace = nmt.NmtWorkspace()
        self.workspace.read_from(filename)



    


class MapsReader(CrossCorrelate):
    def __init__(self, maskA, maskB, masked_on_input_A, masked_on_input_B, binning, nside: int, lmax_sht: int = -1):
        super().__init__(maskA, maskB, masked_on_input_A, masked_on_input_B, binning, nside, lmax_sht = lmax_sht)

    def __call__(self, mappaA: np.ndarray, mappaB: np.ndarray = None, factorA: float = 1, factorB: float = 1, lmax: int = -1):
        mappaB = mappaA if mappaB is None else mappaB
        fA = AlphaNmtField(self.maskA, [mappaA], masked_on_input = self.masked_on_input_A, alpha = factorA, lmax_sht = lmax)
        fB = AlphaNmtField(self.maskB, [mappaB], masked_on_input = self.masked_on_input_B, alpha = factorB, lmax_sht = lmax)
        return super().__call__(fA, fB)
    

    def get_effective_n2_from_counts(self, counts: np.ndarray, mask: np.ndarray = None, weights: np.ndarray = None, alpha: float = 1):
        assert np.allclose(self.maskA, self.maskB), "The two fields must have the same mask as this is for the auto."
        #actually not necessary to have same mask, just they have to be of type galaxy. so maybe in the future we can create a custom type.
            
        if mask is None:
            mask = self.maskA
        else:
            try:
                assert np.allclose(mask, self.maskA)
            except:
                assert np.allclose(mask, self.maskB), "Mask must be either the mask of the first field or the mask of the second field."
                
        return shotutils.get_effective_n2_from_counts(self.workspace, self.coupled_shape, counts, mask, weights)/alpha**2.