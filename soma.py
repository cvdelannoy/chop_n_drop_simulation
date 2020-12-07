from numba import njit
import numpy

def soma(s1, s2, cr, sd):
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

    # option 1: assuming db fingerprint can have no missing restrictions
    for i in range(1, l1+1):
        for j in range(1, l2+1):
            bm = -1 * numpy.inf
            for l in range(max(0, j-3), j):
                    candidate = soma_dist(s1[i-1:i], s2[l:j], cr, sd) + cum_sum[i-1, l]
                    bm = max(candidate, bm)

    # option 2: assuming db fingerprint can have missing restrictions
    # for i in range(1, l1 + 1):
    #     for j in range(1, l2 + 1):
    #         bm = -1 * numpy.inf
    #         for l in range(max(0, j - 5), j):
    #             for k in range(max(0, i - 5), i):
    #                 candidate = soma_dist(s1[k:i], s2[l:j], cr, sd) + cum_sum[k, l]
    #                 bm = max(candidate, bm)
            cum_sum[i, j] = bm
    return cum_sum[1:, 1:]


@njit()
def soma_dist(s1, s2, cr, sd):
    p1 = -1 * cr * abs(s1.shape[0] - s2.shape[0])
    fd = fast_sum(s1) - fast_sum(s2)
    sd2 = s2.shape[0] * sd
    p1 -= fd * fd / (sd2 * sd2)
    return p1


@njit(fastmath=True)
def fast_sum(s):
    acc = 0.
    for x in s: acc += x
    return acc
