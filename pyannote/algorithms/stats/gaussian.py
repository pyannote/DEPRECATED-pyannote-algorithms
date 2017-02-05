#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2012-2016 CNRS

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

# AUTHORS
# Hervé BREDIN - http://herve.niderb.fr

from __future__ import unicode_literals

import numpy as np


class Gaussian(object):

    def __init__(self, covariance_type='full'):

        if covariance_type not in ['full', 'diag']:
            raise ValueError("Invalid value for covariance_type: %s"
                             % covariance_type)

        super(Gaussian, self).__init__()
        self.covariance_type = covariance_type

    def __set_mean(self, mean):
        """Set mean and reset its square"""
        self._mean = mean
        self._mean_square = None

    def __get_mean(self):
        """Get mean"""
        return self._mean

    mean = property(fset=__set_mean, fget=__get_mean)
    """Mean"""

    def __get_mean_square(self):
        """Pre-compute and/or return pre-computed mean square"""
        if self._mean_square is None:
            mean = self.mean.reshape((1, -1))
            self._mean_square = np.dot(mean.T, mean)
        return self._mean_square

    mean_square = property(fget=__get_mean_square)
    """Mean square"""

    def __set_covar(self, covar):
        """Set covariance and reset its inverse & log-determinant"""
        self._covar = covar
        self._inv_covar = None
        self._log_det_covar = None

    def __get_covar(self):
        """Get covariance"""
        return self._covar

    covar = property(fset=__set_covar, fget=__get_covar)
    """Covariance matrix"""

    def __get_inv_covar(self):
        """Pre-compute and/or return pre-computed inverse of covariance"""

        if self._inv_covar is None:
            self._inv_covar = np.linalg.inv(self.covar)
        return self._inv_covar

    inv_covar = property(fget=__get_inv_covar)
    """Inverse of covariance matrix"""

    def __get_log_det_covar(self):
        """Pre-compute and/or return pre-computed log |covar|"""

        if self._log_det_covar is None:
            _, self._log_det_covar = np.linalg.slogdet(self.covar)

        return self._log_det_covar

    log_det_covar = property(fget=__get_log_det_covar)
    """Logarithm of covariance determinant"""

    def fit(self, X):

        # compute gaussian mean
        self.mean = np.mean(X, axis=0).reshape((1, -1))

        # compute gaussian covariance matrix
        if self.covariance_type == 'full':
            self.covar = np.cov(X.T, ddof=0)
        elif self.covariance_type == 'diag':
            self.covar = np.diag(np.diag(np.cov(X.T, ddof=0), k=0))

        # keep track of number of samples
        self.n_samples = len(X)

        return self

    def merge(self, other):

        # number of samples
        n1 = self.n_samples
        n2 = other.n_samples
        n = n1 + n2

        # global gaussian
        g = Gaussian(covariance_type=self.covariance_type)
        g.n_samples = n

        if n1 == 0:
            g.mean = other.mean
            g.covar = other.covar

        elif n2 == 0:
            g.mean = self.mean
            g.covar = self.covar

        else:

            # mean
            m = (n1 * self.mean + n2 * other.mean) / n
            g.mean = m.reshape((1, -1))

            # covariance
            k1 = self.covar
            k2 = other.covar
            k = 1. / n * (n1 * (k1 + self.mean_square) +
                          n2 * (k2 + other.mean_square)) \
                - np.dot(m.T, m)

            # make it diagonal if needed
            if self.covariance_type == 'diag':
                k = np.diag(np.diag(k, k=0))

            g.covar = k

        return g

    def bic(self, other, penalty_coef=3.5, merged=None):

        if merged is None:
            # merge self and other
            g = self.merge(other)
        else:
            g = merged

        delta_bic = bayesianInformationCriterion(
            self, other, g=g, penalty_coef=penalty_coef)

        # return delta bic & merged gaussian
        return delta_bic, g

    def divergence(self, g):
        """
        Gaussian divergence
        """
        dmean = self.mean - g.mean
        return np.float(
            dmean.dot(np.sqrt(self.inv_covar * g.inv_covar)).dot(dmean.T)
        )


class RollingGaussian(Gaussian):

    def __init__(self, covariance_type='full'):
        super(RollingGaussian, self).__init__(covariance_type=covariance_type)
        self.mean = None
        self.covar = None
        self.n_samples = 0
        self.start = 0
        self.end = 0

    def fit(self, X, start=None, end=None):

        if start is None:
            start = 0
        if end is None:
            end = len(X)

        # first call to fit...
        if self.n_samples == 0:
            self.start = start
            self.end = end
            return super(RollingGaussian, self).fit(X[start:end])

        i_index = list(range(start, self.start)) + list(range(self.end, end))
        o_index = list(range(self.start, start)) + list(range(end, self.end))

        i_x = np.take(X, i_index, axis=0)
        o_x = np.take(X, o_index, axis=0)

        n_old = self.n_samples
        n_in = len(i_x)
        n_out = len(o_x)
        n_new = n_old + n_in - n_out

        # estimate new mean

        d = X.shape[1]

        mu_old = self.mean
        mu_in = (np.mean(i_x, axis=0).reshape((1, -1))
                 if n_in else np.zeros((1, d)))
        mu_out = (np.mean(o_x, axis=0).reshape((1, -1))
                  if n_out else np.zeros((1, d)))
        mu_new = ((n_old * mu_old + n_in * mu_in - n_out * mu_out) /
                  (n_old + n_in - n_out))

        # estimate new covariance

        cov_old = self.covar

        cov_in = (np.cov(i_x.T, ddof=0)
                  if n_in else np.zeros((d, d)))
        cov_out = (np.cov(o_x.T, ddof=0)
                   if n_out else np.zeros((d, d)))

        cov_new = (
            (
                n_old * (cov_old + np.dot(mu_old.T, mu_old))
                + n_in * (cov_in + np.dot(mu_in.T, mu_in))
                - n_out * (cov_out + np.dot(mu_out.T, mu_out))
            ) / (n_old + n_in - n_out) - np.dot(mu_new.T, mu_new)
        )

        if self.covariance_type == 'diag':
            cov_new = np.diag(np.diag(cov_new, k=0))

        # remember everything

        self.start = start
        self.end = end
        self.n_samples = n_new

        self.mean = mu_new
        self.covar = cov_new

        return self


def bayesianInformationCriterion(g1, g2, g=None, penalty_coef=1.,
                                 returns_terms=False):
    """Returns Bayesian Information Criterion from 2 Gaussians

    Parameters
    ----------
    g1, g2 : Gaussian
    penalty_coef: float, optional
        Defaults to 1.
    g : Gaussian, optional
        Precomputed merge of g1 and g2
    returns_terms : boolean, optional
        Returns (ratio, penalty) tuple instead of ratio - 𝝀 x penalty
    """

    # merge gaussians if precomputed version is not provided
    if g is None:
        g = g1.merge(g2)

    # number of samples for each gaussian
    n1 = g1.n_samples
    n2 = g2.n_samples
    n = n1 + n2

    # first term of Bayesian information criterion
    ldc = g.log_det_covar if n > 0 else 0.
    ldc1 = g1.log_det_covar if n1 > 0 else 0.
    ldc2 = g2.log_det_covar if n2 > 0 else 0.
    ratio = n * ldc - n1 * ldc1 - n2 * ldc2

    # second term of Bayesian information criterion
    d = g.mean.shape[1]  # number of free parameters
    if g.covariance_type == 'diag':
        n_parameters = 2 * d
    else:
        n_parameters = d + (d * (d + 1)) / 2
    penalty = n_parameters * np.log(n)

    if returns_terms:
        return ratio, penalty

    else:
        return ratio - penalty_coef * penalty
