import numpy as np
from numba import jit

@jit
def normalize(x):
    """
    Amplitude embedding or simply a normalization.
    """
    return x/np.linalg.norm(x)

@jit
def stereo(x):
    """
    Inverse stereoscopic projection embedding.
    """
    return np.append(2*x/(np.linalg.norm(x)**2+1), (np.linalg.norm(x)**2-1)/(np.linalg.norm(x)**2+1))

@jit
def orthogonalize(x):
    """
    Orthogonal embedding (one of our contribution in the paper).
    """
    u = normalize(x)
    u = np.append(u,1.)
    return normalize(u)