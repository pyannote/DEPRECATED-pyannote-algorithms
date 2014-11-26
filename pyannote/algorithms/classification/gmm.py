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

import sklearn
from sklearn.mixture import GMM
from ..utils.sklearn import SKLearnMixin, LabelConverter


def _fit_gmm(base_gmm, X):

    gmm = sklearn.clone(base_gmm)
    return gmm.fit(X)


class SKLearnGMMClassification(object):

    def __init__(self, n_jobs=1, n_components=1, covariance_type='diag',
                 random_state=None, thresh=1e-2, min_covar=1e-3,
                 n_iter=100, n_init=1, params='wmc', init_params='wmc'):

        self.n_components = n_components
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.thresh = thresh
        self.min_covar = min_covar
        self.n_iter = n_iter
        self.n_init = n_init
        self.params = params
        self.init_params = init_params

        self.n_jobs = n_jobs

        super(SKLearnGMMClassification, self).__init__()

    def fit(self, X, y):
        """
        Parameters
        ----------
        X :
        y :
        """

        classes, counts = np.unique(y, return_counts=True)
        priors = 1. * counts / np.sum(counts)

        self.classes_ = classes
        self.prior_ = priors

        K = len(self.classes_)
        assert np.all(self.classes_ == np.arange(K))

        gmm = GMM(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            random_state=self.random_state,
            thresh=self.thresh,
            min_covar=self.min_covar,
            n_iter=self.n_iter,
            n_init=self.n_init,
            params=self.params,
            init_params=self.init_params)

        self.estimators_ = [_fit_gmm(gmm, X[y == i]) for i in self.classes_]

        return self

    def predict_proba(self, X):
        ll = np.array([self.estimators_[i].score(X) for i in self.classes_]).T
        return ll

    def predict(self, X):
        ll = self.predict_proba(X)
        y = np.argmax(ll, axis=1)
        return y


class GMMClassification(SKLearnMixin):

    def __init__(self, n_jobs=1, n_components=1, covariance_type='diag',
                 random_state=None, thresh=1e-2, min_covar=1e-3,
                 n_iter=100, n_init=1, params='wmc', init_params='wmc'):

        self.n_components = n_components
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.thresh = thresh
        self.min_covar = min_covar
        self.n_iter = n_iter
        self.n_init = n_init
        self.params = params
        self.init_params = init_params

        self.n_jobs = n_jobs

    def fit(self, features_iter, annotation_iter):

        self.classifier_ = SKLearnGMMClassification(
            n_jobs=self.n_jobs,
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            random_state=self.random_state,
            thresh=self.thresh,
            min_covar=self.min_covar,
            n_iter=self.n_iter,
            n_init=self.n_init,
            params=self.params,
            init_params=self.init_params
        )

        annotation_iter = list(annotation_iter)
        features_iter = list(features_iter)

        X, y = self.Xy_stack(features_iter, annotation_iter, unknown='unique')

        # convert PyAnnote labels to SKLearn labels
        self.label_converter_ = LabelConverter()
        converted_y = self.label_converter_.fit_transform(y)

        # fit GMM-UBM classifier
        self.classifier_.fit(X, converted_y)

        return self

    def predict_proba(self, features, segmentation):

        # posterior probabilities sklearn-style
        X = self.X(features, unknown='keep')
        ll = self.classifier_.predict_proba(X)

        # convert to pyannote-style & aggregate over each segment
        scores = Scores(uri=segmentation.uri, modality=segmentation.modality)

        sliding_window = features.sliding_window

        for segment, track in segmentation.itertracks():

            # extract ll for all features in segment and aggregate
            i_start, i_duration = sliding_window.segmentToRange(segment)
            p = np.mean(ll[i_start:i_start + i_duration, :], axis=1)
            print p
            print p.shape

            for i, label in enumerate(self.label_converter_):
                scores[segment, track, label] = p[i]

        return scores

    def predict(self, features, segmentation):

        scores = self.predict_proba(features, segmentation)
        return scores.to_annotation(posterior=False)


def _adapt_ubm(ubm, X, adapt_params, adapt_iter):

    # clone UBM (n_components, covariance type, etc...)
    gmm = sklearn.clone(ubm)

    # initialize with UBM precomputed weights, means and covariance matrices
    gmm.n_init = 1
    gmm.init_params = ''
    gmm.weights_ = ubm.weights_
    gmm.means_ = ubm.means_
    gmm.covars_ = ubm.covars_

    # adapt only some parameters
    gmm.params = adapt_params
    gmm.n_iter = adapt_iter
    gmm.fit(X)

    return gmm


class SKLearnGMMUBMClassification(object):

    def __init__(self, n_jobs=1, n_components=1, covariance_type='diag',
                 random_state=None, thresh=1e-2, min_covar=1e-3,
                 n_iter=100, n_init=1, params='wmc', init_params='wmc',
                 precomputed_ubm=None, adapt_iter=10, adapt_params='m'):

        self.n_components = n_components
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.thresh = thresh
        self.min_covar = min_covar
        self.n_iter = n_iter
        self.n_init = n_init
        self.params = params
        self.init_params = init_params

        self.precomputed_ubm = precomputed_ubm  # pre-computed UBM
        self.adapt_iter = adapt_iter
        self.adapt_params = adapt_params

        self.n_jobs = n_jobs

        super(SKLearnGMMUBMClassification, self).__init__()

    def fit_ubm(self, X, y=None):

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

        self.ubm_.fit(X)

        return self.ubm_

    def fit(self, X, y):
        """
        Parameters
        ----------
        X :
        y :
        """

        if self.precomputed_ubm is None:
            self.ubm_ = self.fit_ubm(X, y=y)

        else:
            self.ubm_ = self.precomputed_ubm

        classes, counts = np.unique(y, return_counts=True)
        priors = 1. * counts / np.sum(counts)

        self.open_set_ = (classes[0] == -1)
        if self.open_set_:
            self.classes_ = classes[1:]
            self.prior_ = priors[1:]
            self.unknown_prior_ = priors[0]

        else:
            self.classes_ = classes
            self.prior_ = priors
            self.unknown_prior_ = 0.

        K = len(self.classes_)
        assert np.all(self.classes_ == np.arange(K))

        # TODO assert classes_ = [0, 1, 2, 3, ..., K-1]

        self.estimators_ = [
            _adapt_ubm(self.ubm_, X[y == i],
                       self.adapt_params, self.adapt_iter)
            for i in self.classes_
        ]

        return self

    def _log_likelihood_ratio(self, X):

        ll_ubm = self.ubm_.score(X)
        ll = np.array([self.estimators_[i].score(X) for i in self.classes_]).T
        ll_ratio = (ll.T - ll_ubm).T

        return ll_ratio

    def predict_proba(self, X):

        ll_ratio = self._log_likelihood_ratio(X)

        denominator = (
            self.unknown_prior_ +
            np.exp(logsumexp(ll_ratio, b=self.prior_, axis=1))
        )

        posterior = ((self.prior_ * np.exp(ll_ratio)).T / denominator).T

        return posterior

    def predict(self, X):

        n = X.shape[0]
        y = -np.ones((X.shape[0],), dtype=float)

        posterior = self.predict_proba(X)
        unknown_posterior = 1. - np.sum(posterior, axis=1)

        argmaxima = np.argmax(posterior, axis=1)

        maxima = posterior[range(n), argmaxima]
        known = maxima > unknown_posterior

        y[known] = argmaxima[known]

        return y


class GMMUBMClassification(SKLearnMixin):

    def __init__(self, n_jobs=1, n_components=1, covariance_type='diag',
                 random_state=None, thresh=1e-2, min_covar=1e-3,
                 n_iter=100, n_init=1, params='wmc', init_params='wmc',
                 precomputed_ubm=None, adapt_iter=10, adapt_params='m'):

        self.n_components = n_components
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.thresh = thresh
        self.min_covar = min_covar
        self.n_iter = n_iter
        self.n_init = n_init
        self.params = params
        self.init_params = init_params

        self.precomputed_ubm = precomputed_ubm
        self.adapt_iter = adapt_iter
        self.adapt_params = adapt_params

        self.n_jobs = n_jobs

    def fit(self, features_iter, annotation_iter):

        self.classifier_ = SKLearnGMMUBMClassification(
            n_jobs=self.n_jobs,
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            random_state=self.random_state,
            thresh=self.thresh,
            min_covar=self.min_covar,
            n_iter=self.n_iter,
            n_init=self.n_init,
            params=self.params,
            init_params=self.init_params,
            precomputed_ubm=self.precomputed_ubm,
            adapt_iter=self.adapt_iter,
            adapt_params=self.adapt_params
        )

        annotation_iter = list(annotation_iter)
        features_iter = list(features_iter)

        X, y = self.Xy_stack(features_iter, annotation_iter, unknown='unique')

        # convert PyAnnote labels to SKLearn labels
        self.label_converter_ = LabelConverter()
        converted_y = self.label_converter_.fit_transform(y)

        # fit GMM-UBM classifier
        self.classifier_.fit(X, converted_y)

        return self

    def predict_proba(self, features, segmentation):

        # posterior probabilities sklearn-style
        X = self.X(features, unknown='keep')
        posterior = self.classifier_.predict_proba(X)

        # convert to pyannote-style & aggregate over each segment
        scores = Scores(uri=segmentation.uri, modality=segmentation.modality)

        sliding_window = features.sliding_window

        for segment, track in segmentation.itertracks():

            # extract posterior for all features in segment and aggregate
            i_start, i_duration = sliding_window.segmentToRange(segment)
            p = np.mean(posterior[i_start:i_start + i_duration, :], axis=1)
            print p
            print p.shape

            for i, label in enumerate(self.label_converter_):
                scores[segment, track, label] = p[i]

        return scores

    def predict(self, features, segmentation):

        scores = self.predict_proba(features, segmentation)
        return scores.to_annotation(posterior=True)
