#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2012-2014 CNRS (Hervé BREDIN - http://herve.niderb.fr)

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import unicode_literals

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression


class LLR(object):

    def _get_scores_ratios(self, X, Y, nbins=100):

        finite = np.isfinite(X)
        positive = X[np.where((Y == 1) & finite)]
        negative = X[np.where((Y == 0) & finite)]

        # todo: smarter bins (bayesian blocks)
        # see jakevdp.github.io/blog/2012/09/12/dynamic-programming-in-python/
        m = min(np.min(positive), np.min(negative))
        M = max(np.max(positive), np.max(negative))
        bins = np.arange(m, M, (M - m) / nbins)

        # histograms
        p, _ = np.histogram(positive, bins=bins, density=True)
        n, _ = np.histogram(negative, bins=bins, density=True)

        scores = .5 * (bins[:-1] + bins[1:])
        ratios = np.log(1. * p / n)

        ok = np.where(np.isfinite(ratios))
        scores = scores[ok]
        ratios = ratios[ok]

        # todo: remove bins based on Doddington's "rule of 30"
        # P, _ = np.histogram(positive, bins=bins, density=False)
        # N, _ = np.histogram(negative, bins=bins, density=False)
        # ok = np.where(np.minimum(P, N) > 30)
        # scores = scores[ok]
        # ratios = ratios[ok]

        return scores, ratios

    def _get_prior(self, X, Y):

        positive = X[np.where(Y == 1)]
        negative = X[np.where(Y == 0)]
        return 1. * len(positive) / (len(positive) + len(negative))

    def toPosteriorProbability(self, scores):
        """Get posterior probability given scores

        Parameters
        ----------
        scores : numpy array
            Test scores

        prior : float, optional
            By default, prior is set to the one estimated with .fit()

        Returns
        -------
        posterior : numpy array
            Posterior probability array with same shape as input `scores`

        """

        # Get log-likelihood ratio
        llr = self.toLogLikelihoodRatio(scores)

        # Get prior
        if self.equal_priors:
            prior = 0.5
        else:
            prior = self.prior

        priorRatio = (1. - prior) / prior

        # Compute posterior probability
        return 1 / (1 + priorRatio * np.exp(-llr))


class LLRIsotonicRegression(LLR):
    """Log-likelihood ratio estimation by isotonic regression"""

    def __init__(self, equal_priors=False):
        super(LLRIsotonicRegression, self).__init__()
        self.equal_priors = equal_priors

    def fit(self, X, Y):

        self.prior = self._get_prior(X, Y)

        scores, ratios = self._get_scores_ratios(X, Y)

        y_min = np.min(ratios)
        y_max = np.max(ratios)
        self.ir = IsotonicRegression(y_min=y_min, y_max=y_max)
        self.ir.fit(scores, ratios)

        return self

    def toLogLikelihoodRatio(self, scores):
        """Get log-likelihood ratio given scores

        Parameters
        ----------
        scores : numpy array
            Test scores

        Returns
        -------
        llr : numpy array
            Log-likelihood ratio array with same shape as input `scores`
        """
        x_min = np.min(self.ir.X_)
        x_max = np.max(self.ir.X_)

        oob_min = np.where(scores < x_min)
        oob_max = np.where(scores > x_max)
        ok = np.where((scores >= x_min) * (scores <= x_max))

        calibrated = np.zeros(scores.shape)
        calibrated[ok] = self.ir.transform(scores[ok])
        calibrated[oob_min] = self.ir.y_min
        calibrated[oob_max] = self.ir.y_max
        return calibrated


class LLRLinearRegression(LLR):
    """Log-likelihood ratio estimation by linear regression"""

    def __init__(self, equal_priors=False):
        super(LLRLinearRegression, self).__init__()
        self.equal_priors = equal_priors

    def fit(self, X, Y):

        self.prior = self._get_prior(X, Y)

        scores, ratios = self._get_scores_ratios(X, Y)

        self.lr = LinearRegression(fit_intercept=True, normalize=False)
        self.lr.fit(scores, ratios)

        return self

    def toLogLikelihoodRatio(self, scores):
        """Get log-likelihood ratio given scores

        Parameters
        ----------
        scores : numpy array
            Test scores

        Returns
        -------
        llr : numpy array
            Log-likelihood ratio array with same shape as input `scores`
        """
        return self.lr.transform(scores)


def logsumexp(a, b=None, axis=0):
    """{Over|under}flow-robust computation of log(sum(b*exp(a)))

    Parameters
    ----------
    a : numpy array
    b :
    """
    a = np.rollaxis(a, axis)
    vmax = np.nanmax(a, axis=0)
    if b is None:
        out = np.log(np.sum(np.exp(a - vmax), axis=0))
    else:
        b = np.atleast_2d(b).T
        out = np.log(np.sum(b * np.exp(a - vmax), axis=0))
    out += vmax
    return out
