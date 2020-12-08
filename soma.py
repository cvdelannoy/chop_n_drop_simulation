from numba import njit
import numpy

# @njit(nogil=True)
def soma(s1, s2, cr, sd):
    cum_sum = njit_accumulated_matrix_soma(s1, s2, cr, sd)
    return cum_sum[-1, -1]


# @njit()
def njit_accumulated_matrix_soma(s1, s2, cr, sd, s2_len_variable=True):
    l1 = s1.shape[0]
    l2 = s2.shape[0]
    cum_sum = numpy.full((l1 + 1, l2 + 1), -1 * numpy.inf)
    cum_sum[0, 0] = 0.

    if s2_len_variable:
        # assume db fingerprint can have missing restrictions
        for i in range(1, l1 + 1):
            for j in range(1, l2 + 1):
                bm = -1 * numpy.inf
                for l in range(max(0, j - 2), j):
                    for k in range(max(0, i - 2), i):
                        # if j == l and k == i: pass
                        # elif j - l == k - i: continue
                        candidate = soma_dist(s1[k:i], s2[l:j], cr, sd) + cum_sum[k, l]
                        bm = max(candidate, bm)
                cum_sum[i, j] = bm
    else:
        # assume db fingerprint CANNOT have missing restrictions
        for i in range(1, l1+1):
            for j in range(1, l2+1):
                bm = -1 * numpy.inf
                for l in range(max(0, j-3), j):
                        candidate = soma_dist(s1[i-1:i], s2[l:j], cr, sd) + cum_sum[i-1, l]
                        bm = max(candidate, bm)
    return cum_sum[1:, 1:]


# @njit()
def soma_dist(s1, s2, cr, sd):
    dist = -1 * cr * abs(s1.shape[0] - s2.shape[0])
    dist -= njit_dtw(s1, s2)
    return dist
    # p1 = -1 * cr * abs(s1.shape[0] - s2.shape[0])
    # fd = fast_sum(s1) - fast_sum(s2)
    # sd2 = s2.shape[0] * sd
    # p1 -= fd * fd / (sd2 * sd2)
    # return p1


# @njit(fastmath=True)
def fast_sum(s):
    acc = 0.
    for x in s: acc += x
    return acc


# --- dtw implementation, directly from tslearn ---
# @njit()
def njit_dtw(s1, s2):
    cum_sum = njit_accumulated_matrix(s1, s2)
    return numpy.sqrt(cum_sum[-1, -1])

# @njit()
def njit_accumulated_matrix(s1, s2):
    l1 = s1.shape[0]
    l2 = s2.shape[0]
    cum_sum = numpy.full((l1 + 1, l2 + 1), numpy.inf)
    cum_sum[0, 0] = 0.

    for i in range(l1):
        for j in range(l2):
            cum_sum[i + 1, j + 1] = _local_squared_dist(s1[i], s2[j])
            cum_sum[i + 1, j + 1] += min(cum_sum[i, j + 1],
                                         cum_sum[i + 1, j],
                                         cum_sum[i, j])
    return cum_sum[1:, 1:]

# @njit()
def _local_squared_dist(x, y):
    diff = x - y
    dist = diff * diff
    return dist


# --- dtw_skip implementation ---
# @njit(nogil=True)
def dtw_skip(s1, s2):
    cum_sum = njit_accumulated_matrix_skip(s1, s2)
    return numpy.sqrt(cum_sum[-1, -1])

# @njit()
def njit_accumulated_matrix_skip(s1, s2):
    l1 = s1.shape[0]
    l2 = s2.shape[0]
    cum_sum = numpy.full((l1 + 1, l2 + 1), numpy.inf)
    cum_sum[0, 0] = 0.

    for i in range(l1):
        for j in range(l2):
            ss1 = s1[i] + s1[i-1]
            ss2 = s2[j] + s2[j-1]
            cum_sum[i + 1, j + 1] = min(_local_squared_dist_skip(s1[i], s2[j]) + cum_sum[i, j],
                                        _local_squared_dist_skip(ss1, s2[j]) + cum_sum[i, j+1],
                                        _local_squared_dist_skip(s1[i], ss2) + cum_sum[i+1, j])
    return cum_sum[1:, 1:]

# @njit()
def _local_squared_dist_skip(x, y):
    diff = x - y
    dist = diff * diff
    return dist
