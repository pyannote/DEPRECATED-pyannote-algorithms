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
from ..stats.llr import logsumexp
from pyannote.core import Scores

from sklearn.mixture import GMM
from sklearn.multiclass import OneVsRestClassifier
from ..utils.sklearn_io import SKLearnIOMixin


class _GMM(GMM):

    def fit(self, X, y):
        return super(_GMM, self).fit(X[y == 1])

    def predict_proba(self, X):
        prob = super(_GMM, self).score(X)
        not_used = np.empty(prob.shape)
        return np.vstack([not_used, prob]).T


class _GMMUBM(_GMM):

    def __init__(self, get_ubm=None, n_components=1, covariance_type='diag',
                 random_state=None, thresh=1e-2, min_covar=1e-3,
                 n_iter=100, n_init=1, params='wmc', init_params='wmc',
                 adapt_params='m', adapt_iter=100):

        self.adapt_params = adapt_params
        self.adapt_iter = adapt_iter
        self.get_ubm = get_ubm

        super(_GMMUBM, self).__init__(
            n_components=n_components,
            covariance_type=covariance_type,
            random_state=random_state,
            thresh=thresh,
            min_covar=min_covar,
            n_iter=n_iter,
            n_init=n_init,
            params=params,
            init_params=init_params)

    def fit(self, X, y):

        ubm = self.get_ubm()

        self.weights_ = ubm.weights_
        self.means_ = ubm.means_
        self.covars_ = ubm.covars_

        self.n_init = 1
        self.init_params = ''

        self.params = self.adapt_params
        self.n_iter = self.adapt_iter

        return super(_GMMUBM, self).fit(X, y)


class _GMMClassification(OneVsRestClassifier):

    def fit(self, X, y):

        # TODO: raise an error if 2 classes

        return super(_GMMClassification, self).fit(X, y)


class _GMMUBMClassification(OneVsRestClassifier):

    def _get_ubm(self):
        return self.ubm_

    def __init__(self, n_jobs=1, n_components=1, covariance_type='diag',
                 random_state=None, thresh=1e-2, min_covar=1e-3,
                 n_iter=100, n_init=1, params='wmc', init_params='wmc',
                 adapt_iter=10, adapt_params='m'):

        self.n_components = n_components
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.thresh = thresh
        self.min_covar = min_covar
        self.n_iter = n_iter
        self.n_init = n_init
        self.params = params
        self.init_params = init_params

        self.adapt_iter = adapt_iter
        self.adapt_params = adapt_params

        self.ubm_ = GMM(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            random_state=self.random_state,
            thresh=self.thresh,
            min_covar=self.min_covar,
            n_iter=self.n_iter,
            n_init=self.n_init,
            params=self.params,
            init_params=self.init_params)

        estimator = _GMMUBM(
            get_ubm=self._get_ubm,
            n_components=n_components,
            covariance_type=covariance_type,
            random_state=random_state,
            thresh=thresh,
            min_covar=min_covar,
            n_iter=n_iter,
            n_init=n_init,
            params=params,
            init_params=init_params,
            adapt_iter=adapt_iter,
            adapt_params=adapt_params,
        )

        super(_GMMUBMClassification, self).__init__(estimator, n_jobs=n_jobs)

    def fit(self, X, y):

        # TODO: raise an error if 2 classes

        self.ubm_.fit(X)

        return super(_GMMUBMClassification, self).fit(X, y)

    def log_likelihood_ratio(self, X):

        ll_ubm = self.ubm_.score(X)
        ll = np.array([e.score(X) for e in self.estimators_]).T
        ll_ratio = (ll.T - ll_ubm).T
        return ll_ratio

    def predict_proba(self, X):

        ll_ratio = self.log_likelihood_ratio(X)

        unknown_prior = 0.
        n_classes = len(self.classes_)
        priors = np.ones(n_classes) / n_classes

        denominator = (
            unknown_prior +
            np.exp(logsumexp(ll_ratio, b=priors, axis=1))
        )

        posteriors = ((priors * np.exp(ll_ratio)).T / denominator).T

        return posteriors

    def predict(self, X):

        posteriors = self.predict_proba(X)
        argmaxima = np.argmax(posteriors, axis=1)
        return self.label_binarizer_.classes_[np.array(argmaxima.T)]


class GMMUBMClassification(_GMMUBMClassification, SKLearnIOMixin):

    def train(self, annotation_iter, features_iter):

        X, y = self.Xy_stack(annotation_iter, features_iter)
        self.fit(X, y)

    def apply(self, features, segmentation):
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

        scores = self.scores(features, segmentation)

        if self.open_set:
            # open-set classification returns Unknown
            # when best target score is below unknown prior
            return scores.to_annotation(posterior=True)

        else:
            # close-set classification always returns
            # the target with the best score
            return scores.to_annotation(posterior=False)

    def scores(self, features, segmentation):

        X = self.X(features)

        P = self.predict_proba(X)

        sliding_window = features.sliding_window
        scores = Scores(uri=segmentation.uri, modality=segmentation.modality)

        for segment, track in segmentation.itertracks():

            i0, n = sliding_window.segmentToRange(segment)
            for i, label in self.label_binarizer_.classes_:
                scores[segment, track, label] = np.mean(P[i0:i0 + n, i])

        return scores
