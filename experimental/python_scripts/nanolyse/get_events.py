# -*- coding: utf-8 -*-
"""Example NumPy style docstrings.

This module demonstrates documentation as specified by the `NumPy
Documentation HOWTO`_. Docstrings may extend over multiple lines. Sections
are created with a section header followed by an underline of equal length.

Example
-------
Examples can be given using either the ``Example`` or ``Examples``
sections. Sections support any reStructuredText formatting, including
literal blocks::

    $ python example_numpy.py


Section breaks are created with two blank lines. Section breaks are also
implicitly created anytime a new section starts. Section bodies *may* be
indented:

Notes
-----
    This is an example of an indented section. It's like any other section,
    but the body is indented to help it stand out from surrounding text.

If a section is indented, then a section break is created by
resuming unindented text.

Attributes
----------
module_level_variable1 : int
    Module level variables may be documented in either the ``Attributes``
    section of the module docstring, or in an inline docstring immediately
    following the variable.

    Either form is acceptable, but the two should not be mixed. Choose
    one convention to document module level variables and be consistent
    with it.


.. _NumPy Documentation HOWTO:
   https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt

"""

import numpy as np

def thresholdsearch( signal, sampling_period, levels, dwelltime=0, skip=2, trace=0, t0=0, t1=-1 ):
    """Event detection algorithm using threshold.
    
    This function uses a defined set of cut-off parameters (levels) to determine event locations.
    It utilizes a threshold for shot noise, to prevent preliminary decisions.
    
    Parameters
    ----------
    singal : numpy array
        The signal should be fed as a array of arrays.
        Each top-level array is threated as a trace of a signal allowing easy cross-trace analysis (for e.g. current dependent analysis).
    sampling_period : float
        The sampling_frequency is used in combination with the dwelltime argument exclude short events.
    levels : tuple
        The levels determine the center of the baseline and it's minimal threshold. 
        levels = ( center, threshold ).
    dwelltime : float
        The minimal event length event should have to be included.
    skip : int
        The skip parameter will exclude a number of events between events as shotnoise, and combines them into a single event.
        default = 2.
    trace : int
        The trace to be analysed (n-th number array)
    t0 : int
        The first datapoint to be analysed in the trace.
    t1 : int
        The last datapoint to be analysed in the trace
    
    Returns
    -------
    tuple
        A tuple containing (in order), the baseline events, signal events, baseline events start, basline events end, signal events start and signal events end.
        All start and ends are returned as position of the datapoint in the trace.
        
    """
    try:
        p, l0, l1 = levels
        signal = signal[ trace ][ t0:t1 ]
        n_filter = dwelltime / float( sampling_period ) if dwelltime / float( sampling_period ) > 2 else 2    # While dwelltime suggests that also spikes can be seen, atleast 2 datapoints are required to be an event
        a = np.where( abs( np.array( signal ) ) < ( abs( l0 ) - abs( l1 ) ) )[0]                              # All datapoints above the threshold
        L1_end = np.where( np.diff( a ) > skip )[0]                                                           # Get all event starts (e.g. where two datapoints are maximum 'skip' apart)
        L1_end = np.append( L1_end, len(a)-1 )
        L1_start = np.delete( np.insert(L1_end+1, 0, 0), -1, 0 )                                              # Every end is followed by a new beginning
        idx = np.where( L1_end - L1_start > n_filter )[0]                                                     # Only keep those events that are atleast 2 or n_filter data points long
        L1_start, L1_end = a[ L1_start[ idx ] ], a[ L1_end[ idx ] ]                                           # Set L1 with datapoints that are allowed
        L0_start, L0_end = np.delete( np.insert( L1_end+1, 0, 0 ), -1, 0 ), L1_start                          # Set L0 relative to L1
        L0 = np.array( [ signal[i:j] for i, j in zip( L0_start, L0_end ) ] )                                  # Add the signal of L0
        L1 = np.array( [ signal[i:j] for i, j in zip( L1_start, L1_end ) ] )                                  # Add the signal of L1
        return ( L0, L1, L0_start, L0_end, L1_start, L1_end )
    except:
        pass
