import numpy as np
from scipy.optimize import curve_fit

def get_levels( signal, sigma = 1, trace=0, t0=0, t1=-1 ):
    """Level detection algorithm.
    
    This function uses a gaussian fit around the main signal to determine the open pore current and threshold cut-off.
        
    Parameters
    ----------
    singal : numpy array
        The signal should be fed as a array of arrays.
        Each top-level array is threated as a trace of a signal allowing easy cross-trace analysis (for e.g. current dependent analysis).
    sigma : int
        The number of sigma (multiplier), that the threshold is set from the open pore.
    trace : int
        The trace to be analysed (n-th number array)
    t0 : int
        The first datapoint to be analysed in the trace.
    t1 : int
        The last datapoint to be analysed in the trace
    
    Returns
    -------
    tuple
        A tuple containing (in order), centroid of the open pore current and threshold.
        
    """
    try:
        signal = signal[ trace ][ t0:t1 ]
        mu, std, res = gauss_deconv( abs( np.array( signal ) ) * -1 )
        l0 = mu * np.sign( sum( signal ) ) * - 1
        l1 = ( ( abs( std ) * sigma ) ) * np.sign( sum( signal ) ) * -1
        return ( True, l0, l1 )
    except:
        return ( False, 0, 0 )

def gauss( x, *p ): 
    a, mu, sigma = p
    return a*np.exp( -( x - mu )**2 / float( 2 * sigma**2 ) )

def gauss_deconv( signal, n_peaks = 2 ):
    try:
        bins = np.linspace( min( signal ), max( signal ), 2000 )    # Bins the signal
        hist = np.histogram( np.abs( signal ) * -1, bins=bins )     # Histogram sthe signal
        means, stds = ( [], [] )                                    # Initialise variables
        x = np.diff(hist[1]) + hist[1][0:-1]                        # Calculate the x coordinates
        res = hist[0].astype( 'float64' )                           # Type cast the corresponding y values
        try:
            for i in range( n_peaks ):
                max_bin = np.where( max(res) == res )[0][0]         # Bin width the most counts
                coeff, var_matrix = curve_fit(gauss,                # Fit Gaussian peak
                                              x[max_bin-10:max_bin+10],
                                              res[max_bin-10:max_bin+10],
                                              p0=[ max(res), x[ max_bin ], 1 ])
                res -= gauss(x, *coeff)                             # Subtract the Gaussian peak calculated
                a, mu, s = coeff                                    # Unzip variables
                means.append( mu )
                stds.append( s )
        except:
            pass
        finally:
            if stds[ np.argmax( np.abs( means ) ) ]!=1.0:
                return means[ np.argmax( np.abs( means ) ) ], stds[ np.argmax( np.abs( means ) ) ], res
            else:
                return means[ np.argmax( np.abs( means ) ) ], np.std( signal ), res
    except:
        pass