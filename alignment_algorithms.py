from numba import njit
import numpy as np


def align(s1, s2, algo, gp=None, cr=None, sd2=None, return_trace=False):
    if algo == 'gapped_nw':
        if gp is None:
            raise ValueError('Need gap penalty (gp) for gapped_nw')
        return gapped_nw(s1, s2, gp, return_trace)
    elif algo == 'soma':
        if cr is None or sd2 is None:
            raise ValueError('Need cr and sd2 for soma')
        return soma(s1, s2, cr, sd2, return_trace)
    elif algo == 'soma_alt':
        if cr is None or sd2 is None:
            raise ValueError('Need cr and sd2 for soma_alt')
        return soma_alt(s1, s2, cr, sd2, return_trace)
    elif algo == 'cnd':
        if sd2 is None:
            raise ValueError('Need sd2 for cnd')
        return cnd(s1, s2, sd2, return_trace)
    else:
        raise ValueError(f'{algo} not recognized as algorithm')


def gapped_nw(s1, s2, gp, return_trace=False):
    cs, tp = gapped_nw_njit(s1, s2, gp)
    if return_trace:
        return cs, tp
    return cs[-1,-1]


@njit()
def gapped_nw_njit(s1, s2, gp):
    l1 = s1.shape[0]
    l2 = s2.shape[0]
    cum_sum = np.full((l1 + 1, l2 + 1, 2), np.inf)
    cum_sum[0, 0] = 0.
    tp = np.full((2, l1 + 1, l2 + 1), 0)
    for i in range(l1):
        for j in range(l2):
            steps = np.array([(s1[i] - s2[j]) ** 2 + cum_sum[i, j, 0],
                              (s1[i] + s1[i-1] - s2[j]) ** 2 + cum_sum[i, j+1, 1] + gp,  # for i or j == 0, will not be minimal because skip_mat will be inf
                              (s2[j-1] + s2[j] - s1[i]) ** 2 + cum_sum[i+1, j, 1] + gp])
            wi = np.argmin(steps)
            if wi == 0:
                cum_sum[i+1,j+1, 1] = cum_sum[i, j, 0]
                tp[:, i+1, j+1] = (i,j)
            elif wi == 1:
                cum_sum[i + 1, j + 1, 1] = cum_sum[i, j+1, 0]
                tp[:, i + 1, j + 1] = (i, j+1)
            else:
                cum_sum[i + 1, j + 1, 1] = cum_sum[i+1, j, 0]
                tp[:, i + 1, j + 1] = (i+1, j)
            cum_sum[i+1, j+1, 0] = steps[wi]
    return cum_sum[1:, 1:, 0], tp[:, 1:, 1:] - 1


def cnd(s1, s2, sd2, return_trace=False):
    cs, tp = cnd_njit(s1, s2, sd2, return_trace)
    if return_trace:
        return cs, tp
    return cs[-1,-1]


@njit()
def cnd_njit(s1, s2, sd2, return_trace=False):
    """
    custom alignment procedure for chop-n-drop fingerprints.
    Optionally, return path as 3 x len(s1) x len(s2) matrix. First dim also includes code for gaps:
    0: match
    1: double alignment s2 to s1
    2: gap s1
    3: gap s2
    """
    gp = 1.96 * sd2 ** 0.5 + 500

    pad = np.array([np.inf, np.inf])
    s1 = np.append(pad, s1)
    s2 = np.append(pad, s2)
    l1 = s1.shape[0]
    l2 = s2.shape[0]
    cum_sum = np.full((l1, l2), np.inf)
    cum_sum[1, 1] = 0.
    tp = np.full((3, l1, l2), 0)

    for i in range(2, l1):
        for j in range(2, l2):
            steps = np.array([
                np.abs(s1[i] - s2[j]) + cum_sum[i - 1, j - 1],
                np.abs(s1[i] - s2[j] - s2[j - 1]) + cum_sum[i - 1, j - 2],
                np.abs(s1[i] - s2[j]) + gp + cum_sum[i - 2, j - 1],
                np.abs(s1[i] - s2[j]) + gp + cum_sum[i - 1, j - 2],
            ])
            cum_sum[i,j] = np.min(steps)
            if return_trace:
                wi = np.argmin(steps)
                if wi == 0:
                    tp[:2, i, j] = (i-1, j-1)
                elif wi == 1:
                    tp[:2, i, j] = (i-1, j-2)
                    tp[2, i, j] = 1
                elif wi == 2:
                    tp[:2, i, j] = (i-2, j-1)
                    tp[2, i, j] = 2
                else:
                    tp[:2, i, j] = (i-1, j-2)
                    tp[2, i, j] = 3
    steps = np.array([cum_sum[-1, -1],
                      np.abs(s1[-2] - s2[-1]) + gp + cum_sum[-2, -1],
                      np.abs(s1[-1] - s2[-2]) + gp + cum_sum[-1, -2]
                      ])
    wi = np.argmin(steps)
    cum_sum[-1,-1] = np.min(steps)
    if return_trace:
        if wi == 1:
            tp[:2, -1, -1] = (l1 - 2, l2 - 1)
            tp[2, -1, -1] = 2
        elif wi == 2:
            tp[:2, -1, -1] = (l1 - 1, l2 - 2)
            tp[2, -1, -1] = 3
        tp[:2, :, :] -= 2
    return cum_sum[2:, 2:], tp[:, 2:, 2:]


# --- soma-dtw implementation ---
@njit(nogil=True)
def soma_dtw(s1, s2, cr, sd):
    cum_sum = njit_accumulated_matrix_soma_dtw(s1, s2, cr, sd)
    if cum_sum.shape[1] == 0:
        return -1 * np.inf
    return cum_sum[-1, -1]


@njit()
def njit_accumulated_matrix_soma_dtw(s1, s2, cr, sd, s2_len_variable=True):
    l1 = s1.shape[0]
    l2 = s2.shape[0]
    cum_sum = np.full((l1 + 1, l2 + 1), -1 * np.inf)
    cum_sum[0, 0] = 0.
    # tst = []

    if s2_len_variable:
        # assume db fingerprint can have missing restrictions
        for i in range(1, l1 + 1):
            for j in range(1, l2 + 1):
                bm = -1 * np.inf
                for l in range(max(0, j - 2), j):
                    for k in range(max(0, i - 2), i):
                        # if j == l and k == i: pass
                        # elif j - l == k - i: continue
                        candidate = soma_dtw_dist(s1[k:i], s2[l:j], cr, sd) + cum_sum[k, l]
                        # tst.append(candidate)
                        bm = max(candidate, bm)
                cum_sum[i, j] = bm
                # tst = []
    else:
        # assume db fingerprint CANNOT have missing restrictions
        for i in range(1, l1+1):
            for j in range(1, l2+1):
                bm = -1 * np.inf
                for l in range(max(0, j-3), j):
                        candidate = soma_dtw_dist(s1[i-1:i], s2[l:j], cr, sd) + cum_sum[i-1, l]
                        bm = max(candidate, bm)
    return cum_sum[1:, 1:]


@njit()
def soma_dtw_dist(s1, s2, cr, sd):
    dist = cr * abs(s1.shape[0] - s2.shape[0])  # HIGHER if difference exists
    dist += njit_dtw(s1, s2)  # HIGHER for bigger difference
    return dist


# --- block_punishing soma ---
def soma_alt(s1, s2, cr, sd2, track_path=False):
    cum_sum = njit_accumulated_matrix_soma(s1, s2, cr, sd2, alt=True, track_path=track_path)
    if track_path:
        return cum_sum
    if cum_sum.shape[1] == 0:
        return np.inf
    return cum_sum[-1, -1]


# --- vanilla soma ---
def soma(s1, s2, cr, sd2, track_path=False):
    cum_sum = njit_accumulated_matrix_soma(s1, s2, cr, sd2, alt=False, track_path=track_path)
    if track_path:
        return cum_sum
    if cum_sum.shape[1] == 0:
        return -1 * np.inf
    return cum_sum[-1, -1]


@njit()
def njit_accumulated_matrix_soma(s1, s2, cr, sd2, alt=False, track_path=False):
    """
    :param s1: query fingerprint
    :param s2: database fingerprint
    :param cr: weight of #-of-elements term
    :param sd2: expected sd-squared on value of single query fingerprint value
    :return:
    """
    l1 = s1.shape[0]
    l2 = s2.shape[0]
    cum_sum = np.full((l1 + 1, l2 + 1), np.inf)
    cum_sum[0, 0] = 0.
    if track_path:
        tp = np.full((2, cum_sum.shape[0], cum_sum.shape[1]), 0.0)

    for i in range(1, l1 + 1):
        for j in range(1, l2 + 1):
            bm = np.inf
            for l in range(max(0, j - 2), j):
                for k in range(max(0, i - 2), i):
                    # if j == l and k == i: pass  # allow matching 1 element but disallow matching other equal numbers of elements
                    # elif j - l == k - i: continue
                    candidate = soma_dist(s1[k:i], s2[l:j], cr, sd2, alt) + cum_sum[k, l]
                    if track_path and candidate < bm:
                        cidx = (k,l)
                    bm = min(candidate, bm)
            if track_path: tp[:, i, j] = cidx
            cum_sum[i, j] = bm
    # if track_path:
    #     tp -= 1  # decrease indices as first row and column will be cut
    #     return cum_sum[1:, 1:], tp[:, 1:, 1:]
    return cum_sum[1:, 1:]


@njit()
def soma_dist(s1, s2, cr, sd2, alt=False):
    if alt:
        dist = cr * (s1.shape[0] + s2.shape[0] - 2)  # HIGHER for forming larger blocks
    else:
        dist = cr * abs(s1.shape[0] - s2.shape[0])  # HIGHER for larger difference in number number fragments
    dist += (fast_sum(s1) - fast_sum(s2)) ** 2 / (sd2 * s1.shape[0])  # HIGHER for bigger difference in summed lengths todo: check that s1 is always query fingerprint, not db
    return dist


# --- dtw implementation, directly from tslearn ---
@njit()
def njit_dtw(s1, s2):
    cum_sum = njit_accumulated_matrix(s1, s2)
    return np.sqrt(cum_sum[-1, -1])

@njit()
def njit_accumulated_matrix(s1, s2):
    l1 = s1.shape[0]
    l2 = s2.shape[0]
    cum_sum = np.full((l1 + 1, l2 + 1), np.inf)
    cum_sum[0, 0] = 0.

    for i in range(l1):
        for j in range(l2):
            cum_sum[i + 1, j + 1] = _local_squared_dist(s1[i], s2[j])
            cum_sum[i + 1, j + 1] += min(cum_sum[i, j + 1],
                                         cum_sum[i + 1, j],
                                         cum_sum[i, j])
    return cum_sum[1:, 1:]

@njit()
def _local_squared_dist(x, y):
    diff = x - y
    dist = diff * diff
    return dist


# --- dtw_skip implementation ---
@njit(nogil=True)
def dtw_skip(s1, s2):
    cum_sum = njit_accumulated_matrix_skip(s1, s2)
    return np.sqrt(cum_sum[-1, -1])

@njit()
def njit_accumulated_matrix_skip(s1, s2):
    l1 = s1.shape[0]
    l2 = s2.shape[0]
    cum_sum = np.full((l1 + 1, l2 + 1), np.inf)
    cum_sum[0, 0] = 0.

    for i in range(l1):
        for j in range(l2):
            ss1 = s1[i] + s1[i-1]
            ss2 = s2[j] + s2[j-1]
            cum_sum[i + 1, j + 1] = min(_local_squared_dist_skip(s1[i], s2[j]) + cum_sum[i, j],
                                        _local_squared_dist_skip(ss1, s2[j]) + cum_sum[i, j+1],
                                        _local_squared_dist_skip(s1[i], ss2) + cum_sum[i+1, j])
    return cum_sum[1:, 1:]

@njit()
def _local_squared_dist_skip(x, y):
    diff = x - y
    dist = diff * diff
    return dist


# --- other utilities ---
@njit(fastmath=True)
def fast_sum(s):
    acc = 0.
    for x in s: acc += x
    return acc
