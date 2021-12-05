import numpy as np
import scipy.stats as st


def _divide(a, b):
    try:
        value = a / b
    except ZeroDivisionError:
        value = float('Inf')
    return value


class Connectivity:
    def __init__(self, pre=None, post=None, x1=None, x2=None, y1=None, y2=None, z1=None, z2=None, compute_values=True, compute_sums=True):
        '''
        pre/post are trials of size n_trials x n_bins
        assumes that the stimulation starts on the middle bin
        x1, x2, y1, y2, z1, z2 are bin indices for where
        assumes that there is only one spike per trial, or is interpreted as
        such that in one trial a response is true or false
        '''
        # this gives the possibility to give sums
        if compute_sums:
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
        if compute_values:
            self.compute()

    def compute(self):
        y_refractory = _divide(self.yz_sum, self.z_sum)

        y_response_norefractory = _divide(self.yzinv_sum, self.zinv_sum)

        x_refractory = _divide(self.xz_sum, self.z_sum)

        x_response_norefractory = _divide(self.xzinv_sum, self.zinv_sum)

        y_response_spike = _divide(self.yx_sum, self.x_sum)

        y_nospike = _divide(self.yxinv_sum, self.xinv_sum)

        y0_refractory = _divide(self.y0z_sum, self.z_sum)

        x0_refractory = _divide(self.x0z_sum, self.z_sum)

        y0_response_norefractory = _divide(self.y0zinv_sum, self.zinv_sum)

        x0_response_norefractory = _divide(self.x0zinv_sum, self.zinv_sum)

        y0_response = _divide(self.y0x_sum, self.x_sum)

        y0_nospike = _divide(self.y0xinv_sum, self.xinv_sum)

        if x_refractory =! 0:
            print('Warning: spike window mismatch, spikes in refractory period')
        # standard IV
        self.beta_iv = _divide(
            y_response_norefractory - y_refractory,
            x_response_norefractory - x_refractory)
        # home brew
        self.beta_brew = y_response_spike - y_refractory
        # OLS
        self.beta_ols = y_response_spike - y_nospike

        # DiD iv
        self.beta_iv_did = _divide(
            (y_response_norefractory - y0_response_norefractory) - (y_refractory - y0_refractory),
            (x_response_norefractory - x0_response_norefractory) - (x_refractory - x0_refractory))
        # DiD cace
        self.beta_brew_did = self.beta_brew - (y0_response - y0_refractory)
        # OLS
        self.beta_ols_did = self.beta_ols - (y0_response - y0_nospike)

        self.hit_rate = _divide(self.x_sum, self.n_trials)
