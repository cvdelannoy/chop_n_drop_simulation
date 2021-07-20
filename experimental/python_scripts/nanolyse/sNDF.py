# -*- coding: utf-8 -*-
import numpy as np
from scipy.optimize import curve_fit
from scipy import asarray as ar, exp

# sNDF: super Normal Distribution Function
def sNDF( x, A, x0, sigma, B, C ):
    E = -( ( x - x0 )**2 / ( 2 * sigma**2 ) )**B
    return ( A*exp( E ) ) + C

def fit_sNDF( event_data, sampling_period ):
    L0, L1, L0_start, L0_end, L1_start, L1_end = event_data
    super_fit = []
    super_fit_cov = []
    SD_2 = []
    for i in range( len( L1 ) - 1 ):
        try:
            # Combine the event back with it's surrounding baseline
            Y = np.concatenate( ( L0[i], L1[i], L0[i+1] ) )
            # Get some range of x
            x = np.linspace( 0, len( Y )*sampling_period, len( Y ) )
            # Estimate location of x0
            x0 = ( len( L0[i] ) + len( L1[i] )/2 ) * sampling_period
            # Estimate the sigma
            sigma_x0 = ( len( L1[i] )/2 ) * sampling_period
            # Parameters [amplitude, x0, sigma_x, Beta, offset]
            # Beta=2 -> Normal distribution
            p0 = ( abs( np.mean(L1[i])-np.mean( np.concatenate( ( L0[i], L0[i+1] ) ) ) ), 
                  x0, sigma_x0, 2, np.mean( np.concatenate( ( L0[i], L0[i+1] ) ) ) )
            # Fit a super normal distribution to the event
            popt, pcov = curve_fit( sNDF, x, -1*abs( Y ), p0=p0 )
            SD_2.append( np.average( ( Y - sNDF( x, *popt ) ) ** 2, weights=sNDF( x, *popt ) ) )
            super_fit.append( popt )
            super_fit_cov.append( pcov )
        except:
            pass
    return (super_fit, super_fit_cov, SD_2)

def features_sNDF( super_fit ):
    SF = super_fit
    Iex = np.array([ i[0]/abs(i[-1]) for i in SF ])
    DWT = np.array([ 2 * i[2] * np.sqrt( ( 2 * np.log(2)**(1/i[3]) ) ) for i in SF ])
    beta = np.array([ i[3] for i in SF ])
    return Iex, DWT, beta