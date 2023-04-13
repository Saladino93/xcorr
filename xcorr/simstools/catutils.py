import numpy as np



def poisson_sampling(counts_map: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed = seed)
    return rng.poisson(lam = counts_map)


def delta_g_to_number_counts(delta_g_input_real: np.ndarray, No_observed: int, seed: int, mask: np.ndarray = 1, weights: np.ndarray = 1, alpha: float = 1.) -> np.ndarray:
    
    #this is here so that if I do not want to sample, e.g. because I have a kappa map, I can just return the input map
    if No_observed == 0:
        return delta_g_input_real
    
    mappa = (delta_g_input_real*alpha+1.)
    
    Npix = np.sum(mask) #len(mappa[mask != 0])

    Ngal = mappa*No_observed/Npix
    Ngal = np.nan_to_num(Ngal)*mask*np.nan_to_num(1/weights)/alpha**2.
        
    return (poisson_sampling(Ngal, seed)).astype(int)