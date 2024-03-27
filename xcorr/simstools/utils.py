import itertools
import numpy as np



def divide(a, b):
    return np.divide(a, b, out = np.zeros(a.shape, dtype = float), where = b!= 0)


def get_indices_for_cls_list(N: int):
    '''
    This function returns the indices for the cls list useful for generating spectra in healpy with new order

    Parameters
    ----------
    N : int
        Number of fields
    Returns
    -------
    indices : tuple
        Tuple of two lists, one for the first field and one for the second field
    '''

    xs = [x for x in itertools.chain(*[[i for i in range(N-j)] for j in range(N)])]
    ys = [x for x in itertools.chain(*[[i+j for i in range(N-j)] for j in range(N)])]
    indices = (xs, ys)

    return indices