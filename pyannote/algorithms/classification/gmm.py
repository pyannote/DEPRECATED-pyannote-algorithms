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
from ..stats.lbg import LBG
from ..stats.llr import LLRNaiveBayes, LLRIsotonicRegression, LLRPassthrough
from pyannote.core import Scores

import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.mixture import GMM
from ..utils.sklearn import SKLearnMixin, LabelConverter

from joblib import Parallel, delayed


def fit_gmm_lbg(X, n_components=1, covariance_type='diag',
                random_state=None, thresh=1e-2, min_covar=1e-3,
                n_iter=10, **kwargs):

    lbg = LBG(n_components=n_components, covariance_type=covariance_type,
              random_state=random_state, thresh=thresh, min_covar=min_covar,
              n_iter=n_iter, disturb=0.05, sampling=500)

    gmm = lbg.apply(X)

    return gmm


def fit_gmm(X, n_components=1, covariance_type='diag',
            random_state=None, thresh=1e-2, min_covar=1e-3,
            n_iter=10, n_init=1, params='wmc', init_params='wmc'):

    gmm = GMM(
        n_components=n_components,
        covariance_type=covariance_type,
        random_state=random_state,
        thresh=thresh,
        min_covar=min_covar,
        n_iter=n_iter,
        n_init=n_init,
        params=params,
        init_params=init_params)

    return gmm.fit(X)


def adapt_ubm(ubm, X, adapt_params='m', adapt_iter=10):

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


def fit_naive_bayes(X, y):
    return LLRNaiveBayes().fit(X, y)


def fit_isotonic_regression(X, y):
    return LLRIsotonicRegression().fit(X, y)


def fit_passthrough(X, y):
    return LLRPassthrough().fit(X, y)


class SKLearnGMMClassification(BaseEstimator, ClassifierMixin):

    def __init__(self, n_jobs=1, n_components=1, covariance_type='diag',
                 random_state=None, thresh=1e-2, min_covar=1e-3,
                 n_iter=10, n_init=1, params='wmc', init_params='wmc',
                 calibration=None, lbg=False):

        super(SKLearnGMMClassification, self).__init__()

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

        self.calibration = calibration
        self.lbg = lbg

    def _fit_priors(self, y):

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

    def _fit_estimators(self, X, y):

        if self.lbg:
            fit_func = fit_gmm_lbg
        else:
            fit_func = fit_gmm

        self.estimators_ = Parallel(n_jobs=self.n_jobs)(delayed(fit_func)(
            X[y == k],
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            random_state=self.random_state,
            thresh=self.thresh,
            min_covar=self.min_covar,
            n_iter=self.n_iter,
            n_init=self.n_init,
            params=self.params,
            init_params=self.init_params) for k in self.classes_)

    def _get_fit_calibration(self):

        if self.calibration is None:
            return fit_passthrough

        if self.calibration == 'naive_bayes':
            return fit_naive_bayes

        if self.calibration == 'isotonic':
            return fit_isotonic_regression

    def _fit_calibrations(self, X, y):

        fit_calibration = self._get_fit_calibration()

        raw_scores = self._raw_scores(X)

        self.calibrations_ = Parallel(n_jobs=self.n_jobs)(
            delayed(fit_calibration)(
                raw_scores[:, i],
                np.array(y == k, dtype=int)
            )
            for i, k in enumerate(self.classes_))

    def fit(self, X, y):
        """
        Parameters
        ----------
        X :
        y :
        """

        self._fit_priors(y)
        self._fit_estimators(X, y)
        self._fit_calibrations(X, y)

        return self

    def _raw_scores(self, X):
        return np.array([
            estimator.score(X)
            for estimator in self.estimators_
        ]).T

    def scores(self, X):
        # return estimated log p(X|i) - log p(X|~i) for each each class i

        scores = self._raw_scores(X)
        for i, calibration in enumerate(self.calibrations_):
            scores[:, i] = calibration.transform(scores[:, i])

        return scores

    def predict_log_proba(self, X):

        ll_ratio = self.scores(X)
        prior = self.prior_

        if self.open_set_:
            # append "unknown" prior
            prior = np.hstack([self.prior_, self.unknown_prior_])
            # append "unknown" log-likelihood ratio (zeros)
            zeros = np.zeros((ll_ratio.shape[0], 1))
            ll_ratio = np.hstack([ll_ratio, zeros])

        posterior = ((np.log(prior) + ll_ratio).T -
                     logsumexp(ll_ratio, b=prior, axis=1)).T

        if self.open_set_:
            # remove dimension of unknown prior
            posterior = posterior[:, :-1]

        return posterior

    def predict_proba(self, X):
        return np.exp(self.predict_log_proba(X))

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


class SKLearnGMMUBMClassification(SKLearnGMMClassification):

    def __init__(self, n_jobs=1, n_components=1, covariance_type='diag',
                 random_state=None, thresh=1e-2, min_covar=1e-3,
                 n_iter=10, n_init=1, params='wmc', init_params='wmc',
                 precomputed_ubm=None, adapt_iter=10, adapt_params='m',
                 calibration=None, lbg=False):

        super(SKLearnGMMUBMClassification, self).__init__(
            n_components=n_components, covariance_type=covariance_type,
            random_state=random_state, thresh=thresh, min_covar=min_covar,
            n_iter=n_iter, n_init=n_init, params=params,
            init_params=init_params, calibration=calibration, n_jobs=n_jobs,
            lbg=lbg)

        self.precomputed_ubm = precomputed_ubm
        self.adapt_iter = adapt_iter
        self.adapt_params = adapt_params

    def _fit_ubm_lbg(self, X, y=None):

        self.ubm_ = fit_gmm_lbg(
            X,
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            random_state=self.random_state,
            thresh=self.thresh,
            min_covar=self.min_covar,
            n_iter=self.n_iter,
            n_init=self.n_init)

        return self.ubm_

    def _fit_ubm(self, X, y=None):

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

    def _fit_estimators(self, X, y):

        if self.precomputed_ubm is None:
            if self.lbg:
                self.ubm_ = self._fit_ubm_lbg(X, y=y)
            else:
                self.ubm_ = self._fit_ubm(X, y=y)

        else:
            self.ubm_ = self.precomputed_ubm

        self.estimators_ = Parallel(n_jobs=self.n_jobs)(delayed(adapt_ubm)(
            self.ubm_, X[y == k],
            adapt_params=self.adapt_params,
            adapt_iter=self.adapt_iter) for k in self.classes_)

    def _raw_scores(self, X):
        # should return log-likelihood ratio for each each class
        # log p(X|i) - log p(X|~i) instead of just log p(X|i)
        # here it is approximated as log p(X|i) - log p(X|ω)
        ll = np.array([estimator.score(X) for estimator in self.estimators_]).T
        ll_ubm = self.ubm_.score(X)
        ll_ratio = (ll.T - ll_ubm).T
        return ll_ratio


class GMMClassification(SKLearnMixin):

    def __init__(self, n_jobs=1, n_components=1, covariance_type='diag',
                 random_state=None, thresh=1e-2, min_covar=1e-3,
                 n_iter=10, n_init=1, params='wmc', init_params='wmc',
                 calibration=None, lbg=False):

        self.n_components = n_components
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.thresh = thresh
        self.min_covar = min_covar
        self.n_iter = n_iter
        self.n_init = n_init
        self.params = params
        self.init_params = init_params
        self.calibration = calibration
        self.n_jobs = n_jobs
        self.lbg = lbg

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
            init_params=self.init_params,
            calibration=self.calibration,
            lbg=self.lbg,
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
            p = np.mean(ll[i_start:i_start + i_duration, :], axis=0)

            for i, label in enumerate(self.label_converter_):
                scores[segment, track, label] = p[i]

        return scores

    def predict(self, features, segmentation):

        scores = self.predict_proba(features, segmentation)
        return scores.to_annotation(posterior=False)


class GMMUBMClassification(SKLearnMixin):

    def __init__(self, n_jobs=1, n_components=1, covariance_type='diag',
                 random_state=None, thresh=1e-2, min_covar=1e-3,
                 n_iter=10, n_init=1, params='wmc', init_params='wmc',
                 precomputed_ubm=None, adapt_iter=10, adapt_params='m',
                 calibration=None, lbg=False):

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
        self.calibration = calibration
        self.lbg = lbg

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
            adapt_params=self.adapt_params,
            calibration=self.calibration,
            lbg=self.lbg
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
            p = np.mean(posterior[i_start:i_start + i_duration, :], axis=0)

            for i, label in enumerate(self.label_converter_):
                scores[segment, track, label] = p[i]

        return scores

    def predict(self, features, segmentation):

        scores = self.predict_proba(features, segmentation)
        return scores.to_annotation(posterior=True)
