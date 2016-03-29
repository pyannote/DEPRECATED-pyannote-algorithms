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

"""Linde–Buzo–Gray algorithm"""


import six.moves
import numpy as np
from sklearn.mixture import GMM
import logging


class LBG(object):
    """

    Parameters
    ----------
    n_components : int, optional
        Number of mixture components. Defaults to 1.

    covariance_type : string, optional
        String describing the type of covariance parameters to
        use.  Must be one of 'spherical', 'tied', 'diag', 'full'.
        Defaults to 'diag' (the only one supported for now...)

    random_state: RandomState or an int seed (0 by default)
        A random number generator instance

    min_covar : float, optional
        Floor on the diagonal of the covariance matrix to prevent
        overfitting.  Defaults to 1e-3.

    tol : float, optional
        Convergence threshold. Defaults to 1e-2.

    n_iter : int, optional
        Number of EM iterations per split. Defaults to 10.

    sampling : int, optional
        Reduce the number of samples used for the initialization steps to
        `sampling` samples per component. A few thousands samples per component
        should be a reasonable rule of thumb.
        The final estimation steps always use the whole sample set.

    disturb : float, optional
        Weight applied to variance when splitting Gaussians. Defaults to 0.05.
        mu+ = mu + disturb * sqrt(var)
        mu- = mu - disturb * sqrt(var)

    Attributes
    ----------
    `weights_` : array, shape (`n_components`,)
        This attribute stores the mixing weights for each mixture component.

    `means_` : array, shape (`n_components`, `n_features`)
        Mean parameters for each mixture component.

    `covars_` : array
        Covariance parameters for each mixture component.  The shape
        depends on `covariance_type`::

            (n_components, n_features)             if 'spherical',
            (n_features, n_features)               if 'tied',
            (n_components, n_features)             if 'diag',
            (n_components, n_features, n_features) if 'full'

    `converged_` : bool
        True when convergence was reached in fit(), False otherwise.

    """

    def __init__(self, n_components=1, covariance_type='diag',
                 random_state=None, tol=1e-5, min_covar=1e-3,
                 n_iter=10, disturb=0.05, sampling=0, logger=None):

        if covariance_type != 'diag':
            raise NotImplementedError(
                'Only diagonal covariances are supported.')

        super(LBG, self).__init__()

        self.n_components = n_components
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.tol = tol
        self.min_covar = min_covar
        self.n_iter = n_iter
        self.disturb = disturb
        self.sampling = sampling
        self.logger = logging.getLogger(__name__) if logger is None else logger

    def sample(self, X, n):
        # shuffle-sampling

        N = X.shape[0]

        # keep going
        while True:

            if n == 0 or N <= n:
                yield X
                continue

            # start by shuffling X
            X = np.random.permutation(X)
            for i in six.moves.range(0, N - n, n):
                yield X[i:i + n, :]

    def _split(self, gmm, n_components):
        """Split gaussians and return new mixture.

        Parameters
        ----------
        gmm : sklearn.mixture.GMM
        n_components : int
            Number of components in new mixture with the following constraint:
            gmm.n_components < n_components <= 2 x gmm.n_components

        Returns
        -------
        new_gmm : sklearn.mixture.GMM
            New mixture with n_components components.

        """

        # TODO: sort gmm components in importance order so that the most
        # important ones are the one actually split...

        new_gmm = GMM(n_components=n_components,
                      covariance_type=self.covariance_type,
                      random_state=self.random_state,
                      min_covar=self.min_covar,
                      n_iter=1,
                      params='wmc',
                      n_init=1,
                      init_params='')

        # number of new components to be added
        k = n_components - gmm.n_components

        # split weights
        new_gmm.weights_[:k] = gmm.weights_[:k] / 2
        new_gmm.weights_[k:2 * k] = gmm.weights_[:k] / 2

        # initialize means_ with new number of components
        shape = list(gmm.means_.shape)
        shape[0] = n_components
        new_gmm.means_ = np.zeros(shape, dtype=gmm.means_.dtype)
        # TODO: add support for other covariance_type
        # TODO: for now it only supports 'diag'
        noise = self.disturb * np.sqrt(gmm.covars_[:k, :])
        new_gmm.means_[:k, :] = gmm.means_[:k, :] + noise
        new_gmm.means_[k:2 * k, :] = gmm.means_[:k, :] - noise

        # initialize covars_ with new number of components
        shape = list(gmm.covars_.shape)
        shape[0] = n_components
        new_gmm.covars_ = np.zeros(shape, dtype=gmm.covars_.dtype)
        # TODO: add support for other covariance_type
        # TODO: for now it only supports 'diag'
        new_gmm.covars_[:k, :] = gmm.covars_[:k, :]
        new_gmm.covars_[k:2 * k, :] = gmm.covars_[:k, :]

        # copy remaining unsplit gaussians
        if k < gmm.n_components:
            new_gmm.weights_[2 * k:] = gmm.weights_[k:]
            new_gmm.means_[2 * k:, :] = gmm.means_[k:, :]
            new_gmm.covars_[2 * k:, :] = gmm.covars_[k:, :]

        return new_gmm

    def apply_partial(self, X, gmm=None):

        # initialize GMM with only one gaussian if None is provided
        if gmm is None:
            gmm = GMM(n_components=1, covariance_type=self.covariance_type,
                      random_state=self.random_state,
                      min_covar=self.min_covar, n_iter=1,
                      n_init=1, params='wmc',
                      init_params='')

        previous_ll = -np.inf

        log_splt = "{0} gauss."
        log_iter = (
            "{0} gauss. / iter. #{1} / {2} samples / "
            "llr = {3:.5f} (gain = {4:.5f})"
        )

        while gmm.n_components <= self.n_components:

            self.logger.info(log_splt.format(gmm.n_components))

            # number of samples
            n = self.sampling * gmm.n_components

            # set n to 0 if this is the last iteration
            # so that complete model is trained with all data
            n *= (gmm.n_components < self.n_components)

            # iterate n_iter times (potentially with sampled data)
            for i, x in six.moves.zip(six.moves.range(self.n_iter), self.sample(X, n)):

                # one EM iteration
                gmm.fit(x)

                # compute average log-likelihood gain
                ll = np.mean(gmm.score(X))
                gain = ll - previous_ll

                yield gmm, {'n_components': gmm.n_components,
                            'iteration': i + 1,
                            'log_likelihood': ll}

                # log
                self.logger.debug(log_iter.format(
                    gmm.n_components, i + 1, x.shape[0], ll, gain))

                # converged?
                if (i > 0) and abs(gain) < self.tol:
                    break

                previous_ll = ll

            if gmm.n_components < self.n_components:

                # one EM iteration
                gmm.fit(X)

                # compute average log-likelihood gain
                ll = np.mean(gmm.score(X))
                gain = ll - previous_ll

                yield gmm, {'n_components': gmm.n_components,
                            'iteration': -1,
                            'log_likelihood': ll}

                # log
                self.logger.debug(log_iter.format(
                    gmm.n_components, i + 2, X.shape[0], ll, gain))

            else:
                # stop iterating when requested number of components is reached
                return

            # increase number of components (x 2)
            n_components = min(self.n_components, 2 * gmm.n_components)
            gmm = self._split(gmm, n_components)

    def apply(self, X):
        """Estimate model parameters with LBG initialization and
        the expectation-maximization algorithm.

        Parameters
        ----------
        X : array_like, shape (n, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point.
        """

        for gmm, _ in self.apply_partial(X):
            pass

        return gmm
