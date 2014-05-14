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


"""Linde–Buzo–Gray algorithm"""


import logging
import numpy as np
from sklearn.mixture import GMM


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

    thresh : float, optional
        Convergence threshold. Defaults to 1e-2.

    n_iter : int, optional
        Number of EM iterations per split. Defaults to 10.

    sampling : int, optional
        Reduce the number of samples used for the initialization steps to
        `sampling` samples per component. A few hundreds samples per component
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
                 random_state=None, thresh=1e-2, min_covar=1e-3,
                 n_iter=10, disturb=0.05, sampling=0):

        if covariance_type != 'diag':
            raise NotImplementedError(
                'Only diagonal covariances are supported.')

        super(LBG, self).__init__()

        self.n_components = n_components
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.thresh = thresh
        self.min_covar = min_covar
        self.n_iter = n_iter
        self.disturb = disturb
        self.sampling = sampling

    def _subsample(self, X, n_components):
        """Down-sample data points according to current number of components

        Successive calls will return different sample sets, based on the
        internal _counter which is incremented after each call.

        Parameters
        ----------
        X : array_like, shape (N, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point.

        Returns
        -------
        x : array_like, shape (n < N, n_features)
            Subset of X, with n close to n_components x sampling
        """

        x = X
        step = len(X) / (self.sampling * n_components)
        if step >= 2:
            x = X[(self._counter % step)::step]
            self._counter += 1
        return x

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
                      thresh=self.thresh,
                      min_covar=self.min_covar,
                      n_iter=1,
                      params='wmc',
                      n_init=1,
                      init_params='')

        # number of new components to be added
        k = n_components - gmm.n_components

        # split weights
        new_gmm.weights_[:k] = gmm.weights_[:k] / 2
        new_gmm.weights_[k:2*k] = gmm.weights_[:k] / 2

        # initialize means_ with new number of components
        shape = list(gmm.means_.shape)
        shape[0] = n_components
        new_gmm.means_ = np.zeros(shape, dtype=gmm.means_.dtype)
        # TODO: add support for other covariance_type
        # TODO: for now it only supports 'diag'
        noise = self.disturb * np.sqrt(gmm.covars_[:k, :])
        new_gmm.means_[:k, :] = gmm.means_[:k, :] + noise
        new_gmm.means_[k:2*k, :] = gmm.means_[:k, :] - noise

        # initialize covars_ with new number of components
        shape = list(gmm.covars_.shape)
        shape[0] = n_components
        new_gmm.covars_ = np.zeros(shape, dtype=gmm.covars_.dtype)
        # TODO: add support for other covariance_type
        # TODO: for now it only supports 'diag'
        new_gmm.covars_[:k, :] = gmm.covars_[:k, :]
        new_gmm.covars_[k:2*k, :] = gmm.covars_[:k, :]

        # copy remaining unsplit gaussians
        if k < gmm.n_components:
            new_gmm.weights_[2*k:] = gmm.weights_[k:]
            new_gmm.means_[2*k:, :] = gmm.means_[k:, :]
            new_gmm.covars_[2*k:, :] = gmm.covars_[k:, :]

        return new_gmm

    def apply(self, X):
        """Estimate model parameters with LBG initialization and
        the expectation-maximization algorithm.

        Parameters
        ----------
        X : array_like, shape (n, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point.
        """

        self._counter = 0

        # init with one gaussian
        gmm = GMM(n_components=1, covariance_type=self.covariance_type,
                  random_state=self.random_state, thresh=self.thresh,
                  min_covar=self.min_covar, n_iter=1,
                  n_init=1, params='wmc',
                  init_params='')

        _llr = -np.inf
        while gmm.n_components < self.n_components:

            # fit GMM on a rolling subset of training data
            if self.sampling > 0:

                for i in range(self.n_iter):

                    x = self._subsample(X, gmm.n_components)
                    gmm.fit(x)

                    # --- logging ---------------------------------------------
                    llr = np.mean(gmm.score(X))
                    logging.debug(
                        "%d Gaussians %d frames iter %d llr = %f gain %f" %
                        (gmm.n_components, len(x), i+1, llr, llr-_llr))
                    _llr = llr
                    # ---------------------------------------------------------

            else:

                gmm.n_iter = self.n_iter
                gmm.fit(X)

                # --- logging -------------------------------------------------
                llr = np.mean(gmm.score(X))
                logging.debug(
                    "%d Gaussians %d frames llr = %f gain %f" %
                    (gmm.n_components, len(X), llr, llr-_llr))
                _llr = llr
                # -------------------------------------------------------------

            # increase number of components (x 2)
            n_components = min(self.n_components, 2*gmm.n_components)
            gmm = self._split(gmm, n_components)

        gmm.n_iter = self.n_iter
        gmm.fit(X)

        # --- logging ---------------------------------------------------------
        llr = np.mean(gmm.score(X))
        logging.debug(
            "%d Gaussians %d frames llr = %f gain %f" %
            (gmm.n_components, len(X), llr, llr-_llr))
        _llr = llr
        # ---------------------------------------------------------------------

        return gmm
