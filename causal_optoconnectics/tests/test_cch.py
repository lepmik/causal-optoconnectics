import pytest
import numpy as np
from causal_optoconnectics.cch import correlogram


def test_correlogram():
    t1 = np.arange(0, .5, .1)
    t2 = np.arange(0.1, .6, .1)
    limit = 1
    bin_size = .1
    counts_true = np.array(
        [0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0, 0, 0, 0])
    counts, bins = correlogram(
        t1=t1, t2=t2, bin_size=bin_size, limit=limit, auto=False)
    assert np.array_equal(counts, counts_true)
    # The interpretation of this result is that there are 5 occurences where
    # in the bin 0 to 0.1, i.e.
    idx = np.argmax(counts)
    assert np.allclose((abs(bins[idx - 1]), bins[idx]), (0, 0.1))
    # The correlogram algorithm is identical to, but computationally faster than
    # the histogram of differences of each timepoint, i.e.
    diff = [t2 - t for t in t1]
    counts_slow, bins = np.histogram(diff, bins=bins) # TODO
    assert np.array_equal(counts_slow, counts), (counts_slow, counts, counts_slow.shape, counts.shape)


def test_correlogram_fft():
    t1 = np.arange(0, .5, .1)
    t2 = np.arange(0.1, .6, .1)
    limit = 1
    bin_size = .1
    from causal_optoconnectics.cch import histogram, xcorr
    bins = np.arange(-limit, limit + bin_size, bin_size)
