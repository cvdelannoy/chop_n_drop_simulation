from neo.io import AxonIO
import numpy as np

def loadAxon( fname ):
    """Load data from axon binary file into nanolyse.
    
    Load data from axon binary file into nanolyse.
    
    Parameters
    ----------
    fname : str
        String with the location of the axon binary file to be loaded
    
    Returns
    -------
    Numpy array, float
        Returnsthe signal extracted from the input, and the sampling period as a float
        
    """
    try:
        bl = AxonIO( filename=fname ).read()
        signal = [ np.array( seg.analogsignals[0] )[:,0] for seg in bl[0].segments ] # Signal Trace
        sampling_period = [ np.asscalar( bl[0].segments[0].analogsignals[0].sampling_period ) for seg in bl[0].segments ][0] # Time between datapoints
        return signal, sampling_period
    except:
        print( "Unable to load .abf file" )