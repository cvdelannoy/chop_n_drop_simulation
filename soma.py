from numba import njit
import numpy

def soma(s1, s2, cr, sd):
    # s1 = to_time_series(s1)
    # s2 = to_time_series(s2)
    return njit_soma(s1, s2, cr, sd)

def to_time_series(ts):
    ts_out = ts.copy() if type(ts) == numpy.ndarray else numpy.array(ts)
    if ts_out.ndim <= 1:
        ts_out = ts_out.reshape((-1, 1))
    if ts_out.dtype != numpy.float:
        ts_out = ts_out.astype(numpy.float)
    return ts_out

@njit(nogil=True)
def njit_soma(s1, s2, cr, sd):
    cum_sum = njit_accumulated_matrix_soma(s1, s2, cr, sd)
    return cum_sum[-1, -1]


@njit()
def njit_accumulated_matrix_soma(s1, s2, cr, sd):
    l1 = s1.shape[0]
    l2 = s2.shape[0]
    cum_sum = numpy.full((l1 + 1, l2 + 1), -1 * numpy.inf)
    cum_sum[0, 0] = 0.

    for i in range(l1):
        for j in range(l2):
            bm = -1 * numpy.inf
            for k in range(0, i+1):
                for l in range(0, j+1):
                    candidate = soma_dist(s1[k:i+1], s2[l:j+1], cr, sd) + cum_sum[k, l]
                    bm = max(candidate, bm)
            cum_sum[i + 1, j + 1] = bm
    return cum_sum[1:, 1:]


@njit()
def soma_dist(s1, s2, cr, sd):
    p1 = -1 * cr * abs(s1.shape[0] - s2.shape[0])
    fd = fast_sum(s1) - fast_sum(s2)
    p1 -= fd * fd / (sd * sd)
    return p1


@njit(fastmath=True)
def fast_sum(s):
    acc = 0.
    for x in s: acc += x
    return acc
