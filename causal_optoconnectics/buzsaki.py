import numpy as np
from .tools import hollow_kernel, poisson_continuity_correction
from .cch import correlogram

def cch_convolve(cch, width, hollow_fraction):
    '''Convolve a cross correlation histogram (cch) with a hollow kernel as in
    _[1].
    Parameters
    ----------
    cch : array
        The cross correlation histogram
    width : float
        Width of kernel (std if gaussian)
    hollow_fraction : float
        Fractoin of the central bin to removed.

    Authors
    -------
    Tristan Stoeber, Mikkel Lepperød
    '''
    import scipy.signal as scs
    kernlen = len(cch) - 1
    kernel = hollow_kernel(kernlen, width, hollow_fraction)
    # padd edges
    len_padd = int(kernlen / 2.)
    cch_padded = np.zeros(len(cch) + 2 * len_padd)
    # "firstW/2 bins (excluding the very first bin) are duplicated,
    # reversed in time, and prepended to the cch prior to convolving"
    cch_padded[0:len_padd] = cch[1:len_padd+1][::-1]
    cch_padded[len_padd: - len_padd] = cch
    # # "Likewise, the lastW/2 bins are symmetrically appended to the cch."
    cch_padded[-len_padd:] = cch[-len_padd-1:-1][::-1]
    # convolve cch with kernel
    result = scs.fftconvolve(cch_padded, kernel, mode='valid')
    assert len(cch) == len(result)
    return result


def cch_significance(t1, t2, bin_size, limit, hollow_fraction, width):
    """
    Parameters
    ---------
    t1 : array
        First spiketrain, raw spike times in seconds.
    t2 : array
        Second spiketrain, raw spike times in seconds.
    bin_size : float
        Width of each bar in histogram in seconds.
    limit : float
        Positive and negative extent of histogram, in seconds.
    kernlen : int
        Length of kernel, must be uneven (kernlen % 2 == 1)
    width : float
        Width of kernel (std if gaussian)
    hollow_fraction : float
        Fraction of the central bin to removed.
    References
    ----------
    Stark, E., & Abeles, M. (2009). Unbiased estimation of precise temporal
    correlations between spike trains. Journal of neuroscience methods, 179(1),
    90-100.
    English et al. 2017, Neuron, Pyramidal Cell-Interneuron Circuit Architecture
    and Dynamics in Hippocampal Networks
    Authors
    -------
    Tristan Stoeber, Mikkel Lepperød
    """
    cch, bins = correlogram(
        t1, t2, bin_size=bin_size, limit=limit, density=False)
    pfast = np.zeros(cch.shape)
    cch_smooth = cch_convolve(
        cch=cch, width=width, hollow_fraction=hollow_fraction)
    pfast = poisson_continuity_correction(cch, cch_smooth)
    # ppeak describes the probability of obtaining a peak with positive lag
    # of the histogram, that is signficantly larger than the largest peak
    # in the negative lag direction.
    ppeak = np.zeros(cch.shape)
    max_vals = np.zeros(cch.shape)
    cch_half_len = int(np.floor(len(cch) / 2.))
    max_vals[cch_half_len:] = np.max(cch[:cch_half_len])
    max_vals[:cch_half_len] = np.max(cch[cch_half_len:])
    ppeak = poisson_continuity_correction(cch, max_vals)
    return ppeak, pfast, bins, cch, cch_smooth


def transfer_probability(x, y, bin_size, limit, hollow_fraction, width,
                         y_mu, y_sigma):
    """Calculate the naive transfer probability using a cross correlation
    histogram as in _[1].
    Parameters
    ---------
    x : array
        First spiketrain, raw spike times in seconds.
    y : array
        Second spiketrain, raw spike times in seconds.
    bin_size : float
        Width of each bar in histogram in seconds.
    limit : float
        Positive and negative extent of histogram, in seconds.
    kernlen : int
        Length of kernel, must be uneven (kernlen % 2 == 1)
    width : float
        Width of kernel (std if gaussian)
    hollow_fraction : float
        Fraction of the central bin to removed.
    y_mu : float
        Average time for downstream spikes (y) in response to upstream spikes (x)
    y_sigma : float
        Standard deviation of downstream response times to upstream spikes (x).
    Returns
    -------
    trans_prob : float
    ppeak : float
    pfast : float
    ptime: float
    cmax: float

    References
    ----------
    _[1] : English et al. 2017, Neuron, Pyramidal Cell-Interneuron Circuit Architecture
    and Dynamics in Hippocampal Networks
    Authors
    -------
    Tristan Stoeber, Mikkel Lepperød
    """
    cch, bins = correlogram(
        x, y, bin_size=bin_size, limit=limit, density=False)
    bins = bins[:-1]

    cch_s = cch_convolve(
        cch=cch, width=width, hollow_fraction=hollow_fraction)

    mask = (bins >= y_mu - y_sigma) & (bins <= y_mu + y_sigma)
    cmax = np.max(cch[mask])
    idx, = np.where(cch==cmax * mask)
    idx = idx if len(idx) == 1 else idx[0]
    pfast, = poisson_continuity_correction(cmax, cch_s[idx])
    cch_half_len = int(np.floor(len(cch) / 2.))
    max_pre = np.max(cch[:cch_half_len])
    ppeak, = poisson_continuity_correction(cmax, max_pre)
    ptime = float(bins[idx])
    trans_prob = sum(cch[mask] - cch_s[mask]) / len(x)
    return trans_prob, ppeak, pfast, ptime, cmax
