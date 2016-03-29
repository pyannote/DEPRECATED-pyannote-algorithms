#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2014-2016 CNRS

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

import six
import numpy as np
from ..utils.viterbi import viterbi_decoding, \
    VITERBI_CONSTRAINT_NONE, \
    VITERBI_CONSTRAINT_MANDATORY, \
    VITERBI_CONSTRAINT_FORBIDDEN
from pyannote.core import Annotation, Scores
from pyannote.core.util import pairwise
from ..utils.sklearn import SKLearnMixin, LabelConverter
from ..classification.gmm import \
    SKLearnGMMClassification, SKLearnGMMUBMClassification


class SKLearnGMMSegmentation(SKLearnGMMClassification):
    """

    Parameters
    ----------

    n_components : int, optional
        Number of mixture components. Defaults to 1.

    covariance_type : string, optional
        String describing the type of covariance parameters to
        use.  Must be one of 'spherical', 'tied', 'diag', 'full'.
        Defaults to 'diag'.

    random_state: RandomState or an int seed (None by default)
        A random number generator instance

    min_covar : float, optional
        Floor on the diagonal of the covariance matrix to prevent
        overfitting.  Defaults to 1e-3.

    tol : float, optional
        Convergence threshold.

    n_iter : int, optional
        Number of EM iterations to perform.

    n_init : int, optional
        Number of initializations to perform. the best results is kept

    params : string, optional
        Controls which parameters are updated in the training
        process.  Can contain any combination of 'w' for weights,
        'm' for means, and 'c' for covars.  Defaults to 'wmc'.

    init_params : string, optional
        Controls which parameters are updated in the initialization
        process.  Can contain any combination of 'w' for weights,
        'm' for means, and 'c' for covars.  Defaults to 'wmc'.

    calibration : string, optional
        Controls how log-likelihoods are calibrated into log-likelihood ratios.
        Must be one of 'naive_bayes' (for Gaussian naive Bayes) or 'isotonic'
        for isotonic regression. Defaults to no calibration.

    lbg : boolean, optional
        Controls whether to use the LBG algorithm for training.
        Defaults to False.

    equal_priors : boolean, optional
        Defaults to False
    """

    def _n_classes(self,):
        K = len(self.classes_)
        return K

    def _fit_structure(self, y_iter):

        K = self._n_classes()

        initial = np.zeros((K, ), dtype=float)
        transition = np.zeros((K, K), dtype=float)

        for y in y_iter:

            initial[y[0]] += 1
            for n, m in pairwise(y):
                transition[n, m] += 1

        # log-probabilities
        self.initial_ = np.log(initial / np.sum(initial))
        self.transition_ = np.log(transition.T / np.sum(transition, axis=1)).T

        return self

    def fit(self, X_iter, y_iter):

        y_iter = list(y_iter)

        super(SKLearnGMMSegmentation, self).fit(
            np.vstack([X for X in X_iter]),
            np.hstack([y for y in y_iter]))

        self._fit_structure(y_iter)

        return self

    def predict(self, X, consecutive=None, constraint=None):
        """
        Parameters
        ----------
        X : array-like, shape (N, D)
        consecutive : array-like, shape (K, )
        constraint : array-like, shape (N, K)

        N is the number of samples.
        D is the features dimension.
        K is the number of classes (including the rejection class as the last
        class, when appropriate).

        """

        if self.calibration is None:
            emission = self.predict_log_likelihood(X)
        else:
            emission = self.predict_log_proba(X)

        sequence = viterbi_decoding(
            emission, self.transition_,
            initial=self.initial_,
            consecutive=consecutive, constraint=constraint)

        return sequence


class SKLearnGMMUBMSegmentation(SKLearnGMMUBMClassification):
    """
    Parameters
    ----------

    n_components : int, optional
        Number of mixture components. Defaults to 1.

    covariance_type : string, optional
        String describing the type of covariance parameters to
        use.  Must be one of 'spherical', 'tied', 'diag', 'full'.
        Defaults to 'diag'.

    random_state: RandomState or an int seed (None by default)
        A random number generator instance

    min_covar : float, optional
        Floor on the diagonal of the covariance matrix to prevent
        overfitting.  Defaults to 1e-3.

    tol : float, optional
        Convergence threshold.

    n_iter : int, optional
        Number of EM iterations to perform.

    n_init : int, optional
        Number of initializations to perform. the best results is kept

    params : string, optional
        Controls which parameters are updated in the training
        process.  Can contain any combination of 'w' for weights,
        'm' for means, and 'c' for covars.  Defaults to 'wmc'.

    init_params : string, optional
        Controls which parameters are updated in the initialization
        process.  Can contain any combination of 'w' for weights,
        'm' for means, and 'c' for covars.  Defaults to 'wmc'.

    precomputed_ubm : GMM, optional
        When provided, class GMMs are adapted from this UBM.

    adapt_params : string, optional
        Controls which parameters are updated in the adaptation
        process.  Can contain any combination of 'w' for weights,
        'm' for means, and 'c' for covars.  Defaults to 'm'.

    adapt_iter : int, optional
        Number of EM iterations to perform during adaptation.

    calibration : string, optional
        Controls how raw GMM scores are calibrated into log-likelihood ratios.
        Must be one of 'naive_bayes' (for Gaussian naive Bayes) or 'isotonic'
        for isotonic regression. Defaults to no calibration.

    lbg : boolean, optional
        Controls whether to use the LBG algorithm for training.
        Defaults to False.
    """

    def _n_classes(self,):

        K = len(self.classes_)
        if self.open_set_:
            K = K + 1

        return K

    def _fit_structure(self, y_iter):

        K = self._n_classes()

        initial = np.zeros((K, ), dtype=float)
        transition = np.zeros((K, K), dtype=float)

        for y in y_iter:

            initial[y[0]] += 1
            for n, m in pairwise(y):
                transition[n, m] += 1

        # log-probabilities
        self.initial_ = np.log(initial / np.sum(initial))
        self.transition_ = np.log(transition.T / np.sum(transition, axis=1)).T

        return self

    def fit(self, X_iter, y_iter):

        y_iter = list(y_iter)

        super(SKLearnGMMUBMSegmentation, self).fit(
            np.vstack([X for X in X_iter]),
            np.hstack([y for y in y_iter]))

        self._fit_structure(y_iter)

        return self

    def predict(self, X, consecutive=None, constraint=None):
        """
        Parameters
        ----------
        X : array-like, shape (N, D)
        consecutive : array-like, shape (K, )
        constraint : array-like, shape (N, K)

        N is the number of samples.
        D is the features dimension.
        K is the number of classes (including the rejection class as the last
        class, when appropriate).

        """

        K = self._n_classes()

        N, D = X.shape
        # assert consecutive.shape == (K, )
        # assert constraint.shape == (N, K)

        posteriors = self.predict_proba(X)

        if self.open_set_:
            unknown_posterior = 1. - np.sum(posteriors, axis=1)
            posteriors = np.vstack([posteriors.T, unknown_posterior.T]).T

        sequence = viterbi_decoding(
            np.log(posteriors), self.transition_,
            initial=self.initial_,
            consecutive=consecutive, constraint=constraint)

        if self.open_set_:
            sequence[sequence == (K - 1)] = -1

        return sequence


class GMMSegmentation(SKLearnMixin):
    """

    Parameters
    ----------

    n_components : int, optional
        Number of mixture components. Defaults to 1.

    covariance_type : string, optional
        String describing the type of covariance parameters to
        use.  Must be one of 'spherical', 'tied', 'diag', 'full'.
        Defaults to 'diag'.

    random_state: RandomState or an int seed (None by default)
        A random number generator instance

    min_covar : float, optional
        Floor on the diagonal of the covariance matrix to prevent
        overfitting.  Defaults to 1e-3.

    tol : float, optional
        Convergence threshold.

    n_iter : int, optional
        Number of EM iterations to perform.

    n_init : int, optional
        Number of initializations to perform. the best results is kept

    params : string, optional
        Controls which parameters are updated in the training
        process.  Can contain any combination of 'w' for weights,
        'm' for means, and 'c' for covars.  Defaults to 'wmc'.

    init_params : string, optional
        Controls which parameters are updated in the initialization
        process.  Can contain any combination of 'w' for weights,
        'm' for means, and 'c' for covars.  Defaults to 'wmc'.

    calibration : string, optional
        Controls how raw GMM scores are calibrated into log-likelihood ratios.
        Must be one of 'naive_bayes' (for Gaussian naive Bayes) or 'isotonic'
        for isotonic regression. Defaults to no calibration.

    lbg : boolean, optional
        Controls whether to use the LBG algorithm for training.
        Defaults to False.

    equal_priors : boolean, optional
        Defaults to False.
    """

    def __init__(self, n_jobs=1, n_components=1, covariance_type='diag',
                 random_state=None, tol=1e-2, min_covar=1e-3,
                 n_iter=10, n_init=1, params='wmc', init_params='wmc',
                 calibration=None, lbg=False, equal_priors=False):

        self.n_components = n_components
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.tol = tol
        self.min_covar = min_covar
        self.n_iter = n_iter
        self.n_init = n_init
        self.params = params
        self.init_params = init_params
        self.calibration = calibration
        self.n_jobs = n_jobs
        self.lbg = lbg
        self.equal_priors = equal_priors

    def fit(self, features_iter, annotation_iter):

        self.classifier_ = SKLearnGMMSegmentation(
            n_jobs=self.n_jobs,
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            random_state=self.random_state,
            tol=self.tol,
            min_covar=self.min_covar,
            n_iter=self.n_iter,
            n_init=self.n_init,
            params=self.params,
            init_params=self.init_params,
            calibration=self.calibration,
            lbg=self.lbg,
            equal_priors=self.equal_priors
        )

        X_iter, y_iter = list(zip(*list(
            self.Xy_iter(features_iter, annotation_iter, unknown='unique'))))

        self.label_converter_ = LabelConverter()
        self.label_converter_.fit(np.hstack(y_iter))

        encoded_y_iter = [self.label_converter_.transform(y) for y in y_iter]
        self.classifier_.fit(X_iter, encoded_y_iter)

        return self

    def _constraint(self, constraint, features):

        N = features.getNumber()
        K = self.classifier_._n_classes()

        mapping = self.label_converter_.mapping()
        sliding_window = features.sliding_window

        # defaults to no constraint
        constraint_ = VITERBI_CONSTRAINT_NONE * np.ones((N, K), dtype=int)

        if isinstance(constraint, Scores):

            for segment, _, label, value in constraint.itervalues():
                t, dt = sliding_window.segmentToRange(segment)
                constraint_[t:t + dt, mapping[label]] = value

        if isinstance(constraint, Annotation):

            # forbidden everywhere...
            for label in constraint.labels():
                constraint_[:, mapping[label]] = VITERBI_CONSTRAINT_FORBIDDEN

            # ... but in labeled segments
            for segment, _, label in constraint.itertracks(label=True):
                t, dt = sliding_window.segmentToRange(segment)
                constraint_[t:t + dt, mapping[label]] = \
                    VITERBI_CONSTRAINT_MANDATORY

        return constraint_

    def _consecutive(self, min_duration, features):

        K = self.classifier_._n_classes()
        consecutive = np.ones((K, ), dtype=int)

        sliding_window = features.sliding_window

        if isinstance(min_duration, float):
            consecutive[:] = sliding_window.durationToSamples(min_duration)

        if isinstance(min_duration, dict):
            mapping = self.label_converter_.mapping()
            for label, duration in six.iteritems(min_duration):
                consecutive[mapping[label]] = \
                    sliding_window.durationToSamples(duration)

        return consecutive

    def predict(self, features, min_duration=None, constraint=None):
        """
        Parameters
        ----------
        min_duration : float or dict, optional
            Minimum duration for each label, in seconds.
        constraint : Annotation or Scores, optional
        """

        constraint_ = self._constraint(constraint, features)
        consecutive = self._consecutive(min_duration, features)

        X = self.X(features, unknown='keep')
        sliding_window = features.sliding_window
        converted_y = self.classifier_.predict(
            X, consecutive=consecutive, constraint=constraint_)

        annotation = Annotation()

        diff = list(np.where(np.diff(converted_y))[0])
        diff = [-1] + diff + [len(converted_y)]

        for t, T in pairwise(diff):
            segment = sliding_window.rangeToSegment(t, T - t)
            annotation[segment] = converted_y[t + 1]

        translation = self.label_converter_.inverse_mapping()

        return annotation.translate(translation)

    @classmethod
    def resegment(cls, features, annotation,
                  equal_priors=True, calibration=None,
                  min_duration=None, constraint=None,
                  **segmenter_args):

        segmenter = cls(
            equal_priors=equal_priors,
            calibration=calibration,
            **segmenter_args)

        segmenter.fit([features], [annotation])

        return segmenter.predict(
            features, min_duration=min_duration, constraint=constraint)


class GMMUBMSegmentation(SKLearnMixin):
    """
    Parameters
    ----------

    n_components : int, optional
        Number of mixture components. Defaults to 1.

    covariance_type : string, optional
        String describing the type of covariance parameters to
        use.  Must be one of 'spherical', 'tied', 'diag', 'full'.
        Defaults to 'diag'.

    random_state: RandomState or an int seed (None by default)
        A random number generator instance

    min_covar : float, optional
        Floor on the diagonal of the covariance matrix to prevent
        overfitting.  Defaults to 1e-3.

    tol : float, optional
        Convergence threshold.

    n_iter : int, optional
        Number of EM iterations to perform.

    n_init : int, optional
        Number of initializations to perform. the best results is kept

    params : string, optional
        Controls which parameters are updated in the training
        process.  Can contain any combination of 'w' for weights,
        'm' for means, and 'c' for covars.  Defaults to 'wmc'.

    init_params : string, optional
        Controls which parameters are updated in the initialization
        process.  Can contain any combination of 'w' for weights,
        'm' for means, and 'c' for covars.  Defaults to 'wmc'.

    precomputed_ubm : GMM, optional
        When provided, class GMMs are adapted from this UBM.

    adapt_params : string, optional
        Controls which parameters are updated in the adaptation
        process.  Can contain any combination of 'w' for weights,
        'm' for means, and 'c' for covars.  Defaults to 'm'.

    adapt_iter : int, optional
        Number of EM iterations to perform during adaptation.

    calibration : string, optional
        Controls how raw GMM scores are calibrated into log-likelihood ratios.
        Must be one of 'naive_bayes' (for Gaussian naive Bayes) or 'isotonic'
        for isotonic regression. Defaults to no calibration.

    lbg : boolean, optional
        Controls whether to use the LBG algorithm for training.
        Defaults to False.
    """

    def __init__(self, n_jobs=1, n_components=1, covariance_type='diag',
                 random_state=None, tol=1e-2, min_covar=1e-3,
                 n_iter=10, n_init=1, params='wmc', init_params='wmc',
                 precomputed_ubm=None, adapt_iter=10, adapt_params='m',
                 calibration=None, lbg=False):

        self.n_components = n_components
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.tol = tol
        self.min_covar = min_covar
        self.n_iter = n_iter
        self.n_init = n_init
        self.params = params
        self.init_params = init_params

        self.precomputed_ubm = precomputed_ubm
        self.adapt_iter = adapt_iter
        self.adapt_params = adapt_params

        self.calibration = calibration
        self.lbg = lbg
        self.n_jobs = n_jobs

    def fit(self, features_iter, annotation_iter):

        self.classifier_ = SKLearnGMMUBMSegmentation(
            n_jobs=self.n_jobs,
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            random_state=self.random_state,
            tol=self.tol,
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

        X_iter, y_iter = list(zip(*list(
            self.Xy_iter(features_iter, annotation_iter, unknown='unique'))))

        self.label_converter_ = LabelConverter()
        self.label_converter_.fit(np.hstack(y_iter))

        encoded_y_iter = [self.label_converter_.transform(y) for y in y_iter]
        self.classifier_.fit(X_iter, encoded_y_iter)

        return self

    def _constraint(self, constraint, features):

        N = features.getNumber()
        K = self.classifier_._n_classes()

        mapping = self.label_converter_.mapping()
        sliding_window = features.sliding_window

        constraint_ = VITERBI_CONSTRAINT_NONE * np.ones((N, K), dtype=int)
        if constraint is not None:
            for segment, _, label, value in constraint.itervalues():
                t, dt = sliding_window.segmentToRange(segment)
                constraint_[t:t + dt, mapping[label]] = value

        return constraint_

    def _consecutive(self, min_duration, features):

        K = self.classifier_._n_classes()
        consecutive = np.ones((K, ), dtype=int)

        sliding_window = features.sliding_window

        if isinstance(min_duration, float):
            consecutive[:] = sliding_window.durationToSamples(min_duration)

        if isinstance(min_duration, dict):
            mapping = self.label_converter_.mapping()
            for label, duration in six.iteritems(min_duration):
                consecutive[mapping[label]] = \
                    sliding_window.durationToSamples(duration)

        return consecutive

    def predict(self, features, min_duration=None, constraint=None):
        """
        Parameters
        ----------
        min_duration : float or dict, optional
            Minimum duration for each label, in seconds.
        """

        constraint_ = self._constraint(constraint, features)
        consecutive = self._consecutive(min_duration, features)

        X = self.X(features, unknown='keep')
        sliding_window = features.sliding_window
        converted_y = self.classifier_.predict(
            X, consecutive=consecutive, constraint=constraint_)

        annotation = Annotation()

        diff = list(np.where(np.diff(converted_y))[0])
        diff = [-1] + diff + [len(converted_y)]

        for t, T in pairwise(diff):
            segment = sliding_window.rangeToSegment(t, T - t)
            annotation[segment] = converted_y[t + 1]

        translation = self.label_converter_.inverse_mapping()

        return annotation.translate(translation)
