import numpy as np
from scipy.ndimage.filters import gaussian_filter

def filter_gaussian(signal, sampling_period, Fs):
    """Gaussian filter for signal.
    
    Filters input signal using a gaussian filer and a user defined cut-off frequency.
    
    Parameters
    ----------
    singal : numpy array
        The signal should be fed as a array of arrays.
        Each top-level array is threated as a trace of a signal allowing easy cross-trace analysis (for e.g. current dependent analysis).
    sampling_period : float
        The sampling_frequency is used in combination with the dwelltime argument exclude short events.
    Fs : int
        The cut-off frequency in hertz
    
    Returns
    -------
    Numpy array
        Returns a numpy array, equal dimensions as the input signal
        
    """
    sigma = 1 / ( sampling_period * Fs * 2 * np.pi)
    return gaussian_filter( signal, sigma )