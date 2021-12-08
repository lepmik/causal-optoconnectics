"""
This module contain all connectivity estimates

"""
import numpy as np


def relu(x):
    """Rectifying linear unit.

    Parameters
    ----------
    x : scalar
        Input.

    Returns
    -------
    scalar
        x if x >= 0 else 0.

    """
    return x if x >= 0 else 0.0


class Connectivity:
    def __init__(self, pre=None, post=None, params=None):
        """Compute connectivity estimates.

        Parameters
        ----------
        pre : array (n_trials, n_bins)
            Stimulus trials for putative upstream neuron (the default is None).
            n_bins must be >= x2 - z1. Assumes that there is only one spike
            per trial window, and is interpreted as such that in one trial window
            a response is true or false.
        pre : array (n_trials, n_bins)
            Stimulus trials for putative downstream neuron (the default is None).
            n_bins must be >= 2(y2 - y1). Assumes that there is only one spike
            per trial window, and is interpreted as such that in one trial window
            a response is true or false.
        params : dict
            Temporal parameters, must contain x1, x2, y1, y2, z1, z2
            which are temporal bin indices where x denotes stimulus response
            window for upstream neuron, y the downstream neuron
            and z the refractory period before stimulus (the default is None).
            Assumes z1 < z2 < x1 < x2 & y1 < y2.

        Returns
        -------
        None
            If pre and post are None, the following conditional sums are stored:
            `yz_sum`, `z_sum`, `yzinv_sum`, `zinv_sum`, `yx_sum`, `x_sum`,
            `yxinv_sum`, `xinv_sum`, `xz_sum`, `xzinv_sum`, `y0z_sum`,
            `y0zinv_sum`, `y0x_sum`, `y0xinv_sum`, `x0z_sum`, `x0zinv_sum`,
            where e.g. zinv indicates 1-z.

        Note
        ----
        Six estimates are computed given stimulus induced boolean spike
        indicator `x` of length `n_trials`, i.e. x[0] = 1 means that a spike
        followed the stimulus, downstream spike indicator `y` and a refractory
        indicator `z` where z[1] = 1 means a spike from upstream happended before
        stimulus and consequently x[1] = 0. y0 and x0 are shifted versions of
        y and x respectively.
        ```python
            beta_ols = y[x==1].mean() - y[x==0].mean()
            beta_iv = (
                (y[z==0].mean() - y[z==1].mean()) /
                (x[z==0].mean() - x[z==1].mean())
            )
            # Note that x[z==1].mean() = 0
            beta_brew = y[x==1].mean() - y[z==1].mean(),

            beta_ols_did = (
                y[x==1].mean() - y[x==0].mean() -
                y0[x==1].mean() - y0[x==0].mean()
            )

            beta_iv_did = (
                (y[z==0].mean() - y0[z==0].mean() -
                    (y[z==1].mean() - y0[z==1].mean())) /
                (x[z==0].mean() - x[z==0].mean() + x0[z==1].mean())
            )

            beta_brew_did = (
                y[x==1].mean() - y[z==1].mean() -
                y0[x==1].mean() - y0[z==1].mean()
            )
        ```

        Examples
        --------
        This examples shows how to compute connectivity from pairwise recorded
        neurons A, B and C. Here, A is not connected to C but B is. Since
        A and B are correlated through common stimulus a simple correlation
        would fail to correctly estimate AC=0 and BC=1. Moreover, we add a
        second confounding factor which shuts of A,B and C together, then the
        only correct estimator is the IV method.
        >>> from causal_optoconnectics.tools import compute_trials
        >>> stimulus = np.arange(10, 510, 10).repeat(2).reshape((50, 2)).astype(float)
        >>> stimulus[:,0] = 3 # event id
        >>> A = stimulus + 1
        >>> A[:,0] = 0 # event id
        >>> B = stimulus + 1
        >>> B[:,0] = 1 # event id
        >>> C = stimulus + 3
        >>> C[:,0] = 2 # event id
        >>> # Neuron A only responds 50% of the time because it spikes before
        >>> # stimulus
        >>> A[1::2, 1] -= 1
        >>> # Neuron B only responds 80% of the time, but C follows
        >>> B[::5, 1] -= 1
        >>> C[::5, 1] -= 1
        >>> # Confounding factor shuts of A, B and C together, this makes the
        >>> # OLS estimator fail
        >>> A[::6] = np.nan
        >>> B[::6] = np.nan
        >>> C[::6] = np.nan
        >>> # Estimate hit rate
        >>> idxs = np.arange(50)
        >>> hit_rate_A = 1 - len(np.unique(np.concatenate((idxs[1::2], idxs[::6])))) / 50
        >>> hit_rate_B = 1 - len(np.unique(np.concatenate((idxs[::6], idxs[::5])))) / 50
        >>> # Now we combine the activity together in a event array
        >>> events = np.concatenate([stimulus, A, B, C], 0)
        >>> # Remove nans
        >>> events = events[np.isfinite(events[:,1])]
        >>> sort_idxs = np.argsort(events[:,1], 0)
        >>> events = events[sort_idxs, :].astype(int)
        >>> trials = compute_trials(events, neurons=3, stim_index=3, n1=-2, n2=4)
        >>> trials[0].shape
        (50, 6)
        >>> # Compute connectivity
        >>> # parameters are indices relative to the range n2 - n1,
        >>> # i.e. 6 is the last index in the trial bins
        >>> params = dict(x1=3, x2=4, y1=5, y2=6, z1=1, z2=3)
        >>> AC = Connectivity(pre=trials[0], post=trials[2], params=params)
        >>> BC = Connectivity(pre=trials[1], post=trials[2], params=params)
        >>> # First we look at the raw values
        >>> AC.compute(rectify=False)
        >>> BC.compute(rectify=False)
        >>> # The IV estimate is negatively biased
        >>> round(AC.beta_iv, 3)
        -0.438
        >>> # The IV estimate is computed by
        >>> # (yzinv_sum / zinv_sum - yz_sum / z_sum) / (xzinv_sum / zinv_sum)
        >>> # where z are trials where the upstream neuron is refractory and
        >>> # zinv are trials where it's not. These latter trials are not
        >>> # very instructive since the fact that it is refractory does not
        >>> # necessarily imply that it spikes. If we look at these numbers it
        >>> # becomes clear why we get a negative estimate
        >>> AC.yzinv_sum, AC.zinv_sum, AC.yz_sum, AC.z_sum, AC.xzinv_sum
        (13, 25, 20, 25, 16)
        >>> # The IV estimate is (13 / 25 - 20 / 25) / (16 / 25)
        >>> #
        >>> # The OLS-IV brew has a positve bias
        >>> round(AC.beta_brew, 3)
        0.012
        >>> # The OLS method is severly off
        >>> round(AC.beta_ols, 3)
        0.224
        >>> # The DiD correction decrease the negative bias for the IV estimate
        >>> round(AC.beta_iv_did, 3)
        -0.122
        >>> # The OLS-IV brew with DiD increases the positve bias
        >>> round(AC.beta_brew_did, 3)
        0.025
        >>> # The DiD correction helps the OLS estimate
        >>> round(AC.beta_ols_did, 3)
        0.184
        >>> round(AC.hit_rate, 2) == round(hit_rate_A, 2)
        True
        >>> round(AC.hit_rate, 2) == round(hit_rate_A, 2)
        True
        >>> #
        >>> # Since we know that the connection is excitatory we can rectify
        >>> AC.compute(rectify=True)
        >>> BC.compute(rectify=True)
        >>> # The IV estimate is negatively biased
        >>> AC.beta_iv
        0.0
        >>> AC.beta_iv_did
        0.0

        """
        # this gives the possibility to only give sums
        if pre is not None and post is not None:
            x1, x2, y1, y2, z1, z2 = map(
                params.get, ['x1', 'x2', 'y1', 'y2', 'z1', 'z2'])
            self.n_trials, n_bins = pre.shape
            assert n_bins % 2 == 0
            n_response_y = y2-y1
            n_response_x = x2-x1

            x = pre[:, x1:x2].sum(1).astype(bool)
            y = post[:, y1:y2].sum(1).astype(bool)
            z = pre[:, z1:z2].sum(1).astype(bool)

            # for DiD
            y0 = post[:,y1-n_response_y:y2-n_response_y].sum(1).astype(bool)
            x0 = pre[:,x1-n_response_x:x2-n_response_x].sum(1).astype(bool)

            self.yz_sum = (y*z).sum()
            self.z_sum = z.sum()
            self.yzinv_sum = (y*(1-z)).sum()
            self.zinv_sum = (1-z).sum()
            self.yx_sum = (y*x).sum()
            self.x_sum = x.sum()
            self.yxinv_sum = (y*(1-x)).sum()
            self.xinv_sum = (1-x).sum()
            self.xz_sum = (x*z).sum()
            self.xzinv_sum = (x*(1-z)).sum()
            self.y0z_sum = (y0*z).sum()
            self.y0x_sum = (y0*x).sum()
            self.y0zinv_sum = (y0*(1-z)).sum()
            self.y0xinv_sum = (y0*(1-x)).sum()
            self.x0z_sum = (x0*z).sum()
            self.x0zinv_sum = (x0*(1-z)).sum()

    def compute(self, rectify=False):
        """Computes connectivity estimates.

        Returns
        -------
        None
            Stores connectivity estimates `beta_iv`, `beta_ols`, `beta_brew`,
            `beta_iv_did`, `beta_ols_did`, `beta_brew_did`.
        """
        y_refractory = np.divide(self.yz_sum, self.z_sum)

        y_response_norefractory = np.divide(self.yzinv_sum, self.zinv_sum)

        # x_refractory = np.divide(self.xz_sum, self.z_sum) # this is set to zero and asserted below

        x_response_norefractory = np.divide(self.xzinv_sum, self.zinv_sum)

        y_response_spike = np.divide(self.yx_sum, self.x_sum)

        y_nospike = np.divide(self.yxinv_sum, self.xinv_sum)

        y0_refractory = np.divide(self.y0z_sum, self.z_sum)

        x0_refractory = np.divide(self.x0z_sum, self.z_sum)

        y0_response_norefractory = np.divide(self.y0zinv_sum, self.zinv_sum)

        x0_response_norefractory = np.divide(self.x0zinv_sum, self.zinv_sum)

        y0_response = np.divide(self.y0x_sum, self.x_sum)

        y0_nospike = np.divide(self.y0xinv_sum, self.xinv_sum)

        if self.xz_sum != 0:
            print('Warning: spike window mismatch, spikes in refractory period')
        # standard IV
        self.beta_iv = np.divide(
            y_response_norefractory - y_refractory,
            x_response_norefractory)
        # home brew
        self.beta_brew = y_response_spike - y_refractory
        # OLS
        self.beta_ols = y_response_spike - y_nospike

        # DiD iv
        self.beta_iv_did = np.divide(
            (y_response_norefractory - y0_response_norefractory) - (y_refractory - y0_refractory),
            (x_response_norefractory - x0_response_norefractory) + x0_refractory) # - (x_refractory - x0_refractory)
        # DiD brew
        self.beta_brew_did = self.beta_brew - (y0_response - y0_refractory)
        # OLS DiD: y_response_spike - y0_response - (y_nospike - y0_nospike)
        # y_response_spike - y0_response - y_nospike + y0_nospike
        # y_response_spike - y_nospike - (y0_response - y0_nospike)
        self.beta_ols_did = self.beta_ols - (y0_response - y0_nospike)
        #
        if rectify:
            self.beta_iv = relu(self.beta_iv)
            self.beta_ols = relu(self.beta_ols)
            self.beta_brew = relu(self.beta_brew)
            self.beta_iv_did = relu(self.beta_iv_did)
            self.beta_ols_did = relu(self.beta_ols_did)
            self.beta_brew_did = relu(self.beta_brew_did)

        self.hit_rate = np.divide(self.x_sum, self.n_trials)
