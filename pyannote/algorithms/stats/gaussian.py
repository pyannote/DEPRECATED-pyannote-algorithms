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

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import unicode_literals

import numpy as np


class Gaussian(object):

    def __init__(self, covariance_type='full'):

        if covariance_type not in ['full', 'diag']:
            raise ValueError("Invalid value for covariance_type: %s"
                             % covariance_type)

        super(Gaussian, self).__init__()
        self.covariance_type = covariance_type

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

        # if it is computed already, returns it
        if self._inv_covar is not None:
            return self._inv_covar

        # otherwise, we need to compute and store it before returning it
        self._inv_covar = np.linalg.inv(self.covar)
        return self._inv_covar

    inv_covar = property(fget=__get_inv_covar)
    """Inverse of covariance matrix"""

    def __get_log_det_covar(self):
        """Pre-compute and/or return pre-computed log |covar|"""

        # if it is computed already, returns it
        if self._log_det_covar is not None:
            return self._log_det_covar

        # otherwise, we need to compute and store it before returning it
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
            m1 = self.mean.reshape((1, -1))
            m2 = other.mean.reshape((1, -1))
            m = (n1*m1+n2*m2)/n
            g.mean = m

            # covariance
            k1 = self.covar
            k2 = other.covar
            k = 1./n * (n1*(k1+np.dot(m1.T, m1)) +
                        n2*(k2+np.dot(m2.T, m2))) \
                - np.dot(m.T, m)

            # make it diagonal if needed
            if self.covariance_type == 'diag':
                k = np.diag(np.diag(k, k=0))

            g.covar = k

        return g

    def bic(self, other, penalty_coef=3.5):

        # merge self and other
        g = self.merge(other)

        # number of free parameters
        d, _ = g.covar.shape
        if g.covariance_type == 'full':
            N = int(d*(d+1)/2. + d)
        elif g.covariance_type == 'diag':
            N = 2*d

        # compute delta BIC
        n = g.n_samples
        n1 = self.n_samples
        n2 = other.n_samples


        if n == 0:
            ldc = 0.
        else:
            ldc = g.log_det_covar

        if n1 == 0:
            ldc1 = 0.
        else:
            ldc1 = self.log_det_covar

        if n2 == 0:
            ldc2 = 0.
        else:
            ldc2 = other.log_det_covar

        delta_bic = n*ldc - n1*ldc1 - n2*ldc2 - penalty_coef*N*np.log(n)

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
