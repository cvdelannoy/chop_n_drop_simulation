# -*- coding: utf-8 -*-
import numpy as np

def get_features_THS( event_data, sampling_period ):
    L0, L1, L0_start, L0_end, L1_start, L1_end = event_data
    L0_mean = np.array([ np.mean( i ) for i in L0 ])
    L0_SD = np.array([ np.std( i ) for i in L0 ])
    L1_mean = np.array([ np.mean( i ) for i in L1 ])
    L1_SD = np.array([ np.std( i ) for i in L1 ])

    # Calculate the residual currents in each events
    Ires_L1 = np.array([ i / float( j ) for i, j in zip( L1, L0_mean ) ])
    
    # Calculate the residual current (Ires), Ires variance and dwelltime (in seconds)
    Ires = L1_mean / L0_mean
    Ires_SD_2 = ( ( L1_mean / L0_mean )**2 ) * ( ( ( L1_SD**2 ) / ( L1_mean**2 ) ) + ( ( L0_SD**2 ) / ( L0_mean**2 ) ) )
    dwelltime = np.array( [ len( i ) * sampling_period for i in Ires_L1 ] )
    return (1-Ires), Ires_SD_2, dwelltime