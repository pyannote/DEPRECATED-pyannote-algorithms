#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2014 CNRS (Hervé BREDIN - http://herve.niderb.fr)

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
import logging
from ..stats.lbg import LBG
from ..stats.llr import logsumexp
from base import BaseClassification
import sklearn
from pyannote.core.annotation import Unknown


class GMMClassification(BaseClassification):
    """GMM-based classification (one GMM per target)

    targets : iterable, optional
        When provided, targets contain the list of target to be recognized.
        All other labels encountered during training are considered as unknown.

    equal_priors : bool, optional
        When False, use learned priors. Defaults to True (equal priors).

    open_set : bool, optional
        When True, perform open-set classification
        Defaults to False (close-set classification).

    == Gaussian Mixture Models ==

    n_components : int, optional
        Number of mixture components in GMM. Defaults to 1.

    covariance_type : string, optional
        String describing the type of covariance parameters to
        use.  Must be one of 'spherical', 'tied', 'diag', 'full'.
        Defaults to 'diag' (the only one supported for now...)

    n_iter : int, optional
        Number of EM iterations to perform during training.
        Defaults to 10.

    random_state: RandomState or an int seed (0 by default)
        A random number generator instance

    min_covar : float, optional
        Floor on the diagonal of the covariance matrix to prevent
        overfitting.  Defaults to 1e-3.

    thresh : float, optional
        Convergence threshold. Defaults to 1e-2.

    sampling : int, optional
        Reduce the number of samples used for the initialization steps to
        `sampling` samples per component. A few hundreds samples per component
        should be a reasonable rule of thumb.
        The final estimation steps always use the whole sample set.

    disturb : float, optional
        Weight applied to variance when splitting Gaussians. Defaults to 0.05.
        mu+ = mu + disturb * sqrt(var)
        mu- = mu - disturb * sqrt(var)

    balance : bool, optional
        If True, try to balance target durations used for training of the UBM.
        Defaults to False (i.e. use all available data).

    n_jobs : int, optional
        Number of parallel jobs for GMM adaptation
        (default is one core). Use -1 for all cores.


    """

    def __init__(self, n_components=1, covariance_type='diag',
                 random_state=None, thresh=1e-2, min_covar=1e-3, n_iter=10,
                 disturb=0.05, sampling=0, balance=False, targets=None,
                 params='m', equal_priors=True, open_set=False, n_jobs=1):

        self.n_components = n_components
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.thresh = thresh
        self.min_covar = min_covar
        self.n_iter = n_iter
        self.disturb = disturb
        self.sampling = sampling
        self.balance = balance
        self.targets = targets
        self.params = params
        self.equal_priors = equal_priors
        self.open_set = open_set
        self.n_jobs = n_jobs
        self.targets = targets

        self._lbg = LBG(n_components=self.n_components,
                        covariance_type=self.covariance_type,
                        random_state=self.random_state, thresh=self.thresh,
                        min_covar=self.min_covar, n_iter=self.n_iter,
                        disturb=self.disturb, sampling=self.sampling)

    def _fit_model(self, data):
        gmm = self._lbg.apply(data)
        return gmm

    def _apply_model(self, target_model, data):
        target_scores = target_model.score(data)
        return target_scores

    def predict(self, segmentation, features):
        """Predict label of each track

        Parameters
        ----------
        segmentation : pyannote.Annotation
            Pre-computed segmentation.
        features : pyannote.SlidingWindowFeature
            Pre-computed features.

        Returns
        -------
        prediction : pyannote.Annotation
            Copy of `segmentation` with predicted labels (or Unknown).

        """

        scores = self.scores(segmentation, features)
        return scores.to_annotation()


class GMMUBMClassification(GMMClassification):
    """GMM/UBM speaker identification

    This is an implementation of the Universal Background Model adaptation
    technique usually applied in the speaker identification community.

    targets : iterable, optional
        When provided, targets contain the list of target to be recognized.
        All other labels encountered during training are considered as unknown.

    == Universal Background Model ==

    n_components : int, optional
        Number of mixture components in UBM. Defaults to 1.

    covariance_type : string, optional
        String describing the type of covariance parameters to
        use.  Must be one of 'spherical', 'tied', 'diag', 'full'.
        Defaults to 'diag' (the only one supported for now...)

    n_iter : int, optional
        Number of EM iterations to perform during training/adaptation.
        Defaults to 10.

    random_state: RandomState or an int seed (0 by default)
        A random number generator instance

    min_covar : float, optional
        Floor on the diagonal of the covariance matrix to prevent
        overfitting.  Defaults to 1e-3.

    thresh : float, optional
        Convergence threshold. Defaults to 1e-2.

    sampling : int, optional
        Reduce the number of samples used for the initialization steps to
        `sampling` samples per component. A few hundreds samples per component
        should be a reasonable rule of thumb.
        The final estimation steps always use the whole sample set.

    disturb : float, optional
        Weight applied to variance when splitting Gaussians. Defaults to 0.05.
        mu+ = mu + disturb * sqrt(var)
        mu- = mu - disturb * sqrt(var)

    balance : bool, optional
        If True, try to balance target durations used for training of the UBM.
        Defaults to False (i.e. use all available data).

    == Adaptation ==

    params : string, optional
        Controls which parameters are adapted.  Can contain any combination
        of 'w' for weights, 'm' for means, and 'c' for covars.
        Defaults to 'm'.

    n_iter : int, optional
        Number of EM iterations to perform during training/adaptation.
        Defaults to 10.

    n_jobs : int, optional
        Number of parallel jobs for GMM adaptation
        (default is one core). Use -1 for all cores.

    == Scoring ==

    equal_priors : bool, optional
        When False, use learned priors. Defaults to True (equal priors).

    open_set : bool, optional
        When True, perform open-set classification
        Defaults to False (close-set classification).

    """

    def __init__(self, n_components=1, covariance_type='diag',
                 random_state=None, thresh=1e-2, min_covar=1e-3, n_iter=10,
                 disturb=0.05, sampling=0, balance=False, targets=None,
                 params='m', equal_priors=True, open_set=False, n_jobs=1):

        self.n_components = n_components
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.thresh = thresh
        self.min_covar = min_covar
        self.n_iter = n_iter
        self.disturb = disturb
        self.sampling = sampling
        self.balance = balance
        self.targets = targets
        self.params = params
        self.equal_priors = equal_priors
        self.open_set = open_set
        self.n_jobs = n_jobs
        self.targets = targets

        self._lbg = LBG(n_components=self.n_components,
                        covariance_type=self.covariance_type,
                        random_state=self.random_state, thresh=self.thresh,
                        min_covar=self.min_covar, n_iter=self.n_iter,
                        disturb=self.disturb, sampling=self.sampling)

    def _fit_background(self, data):
        ubm = self._lbg.apply(data)
        return ubm

    def _adapt_ubm(self, data):
        """Adapt UBM to new data using the EM algorithm

        Parameters
        ----------
        data : array_like, shape (n, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point.

        Returns
        -------
        gmm : GMM
            Adapted UBM

        """
        if not hasattr(self, '_background'):
            raise RuntimeError(
                "Missing background model. Use 'prefit' first.")

        # copy UBM structure and parameters
        gmm = sklearn.clone(self._background)
        gmm.params = self.params  # only adapt requested parameters
        gmm.n_iter = self.n_iter
        gmm.n_init = 1
        gmm.init_params = ''      # initialize with UBM attributes

        # initialize with UBM attributes
        gmm.weights_ = self._background.weights_
        gmm.means_ = self._background.means_
        gmm.covars_ = self._background.covars_

        # --- logging ---------------------------------------------------------
        _llr = np.mean(gmm.score(data))
        logging.debug("llr before adaptation = %f" % _llr)
        # ---------------------------------------------------------------------

        # adaptation
        try:
            gmm.fit(data)
        except ValueError, e:
            logging.error(e)

        # --- logging ---------------------------------------------------------
        llr = np.mean(gmm.score(data))
        logging.debug(
            "llr after adaptation = %f, gain = %f" % (llr, llr - _llr))
        # ---------------------------------------------------------------------

        return gmm

    def _fit_model(self, data):
        gmm = self._adapt_ubm(data)
        return gmm

    def _apply_background(self, data, targets_scores):
        background_scores = self._background.score(data)
        return (targets_scores.T - background_scores).T

    def _llr2posterior(self, llr, priors, unknown_prior):
        denominator = (
            unknown_prior +
            np.exp(logsumexp(llr, b=priors, axis=1))
        )
        posteriors = ((priors * np.exp(llr)).T / denominator).T
        return posteriors

    def predict_proba(self, segmentation, features):
        """Compute posterior probabilities

        Parameters
        ----------
        segmentation : Annotation
            Pre-computed segmentation.
        features : pyannote.SlidingWindowFeature
            Pre-computed features.

        Returns
        -------
        probs : Scores
            For each (segment, track) in `segmentation`, `scores` provides
            the posterior probability for each class.

        """

        # get raw log-likelihood ratio
        scores = self.scores(segmentation, features)

        # reduce Unknown prior to 0. in case of close-set classification
        unknown_prior = self._prior.get(Unknown, 0.)
        if self.open_set is False:
            unknown_prior = 0.

        # number of known targets
        n_targets = len(self.targets)

        if self.equal_priors:

            # equally distribute known prior between known targets
            priors = (1 - unknown_prior) * np.ones(n_targets) / n_targets

        else:

            # ordered known target priors
            priors = np.array([self._prior[t] for t in self.targets])

            # in case of close-set classification
            # equally distribute unknown prior to known targets
            if self.open_set is False:
                priors = priors + self._prior.get(Unknown, 0.) / n_targets

        # compute posterior from LLR directly on the internal numpy array
        func = lambda llr: self._llr2posterior(llr, priors, unknown_prior)
        return scores.apply(func)

    def predict(self, segmentation, features):
        """Predict label of each track

        Parameters
        ----------
        segmentation : pyannote.Annotation
            Pre-computed segmentation.
        features : pyannote.SlidingWindowFeature
            Pre-computed features.

        Returns
        -------
        prediction : pyannote.Annotation
            Copy of `segmentation` with predicted labels (or Unknown).

        """

        probs = self.predict_proba(segmentation, features)

        if self.open_set:
            # open-set classification returns Unknown
            # when best target score is below unknown prior
            return probs.to_annotation(posterior=True)

        else:
            # close-set classification always returns
            # the target with the best score
            return probs.to_annotation(posterior=False)
