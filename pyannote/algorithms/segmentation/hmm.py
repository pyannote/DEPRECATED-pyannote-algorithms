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
import itertools
from ..stats.lbg import LBG
from ..utils.viterbi import viterbi_decoding, VITERBI_CONSTRAINT_NONE
from pyannote.core import Annotation
from pyannote.core.util import pairwise
import sklearn
from ..utils.sklearn import SKLearnMixin, LabelConverter
from ..classification.gmm import SKLearnGMMClassification, SKLearnGMMUBMClassification


class SKLearnGMMSegmentation(SKLearnGMMClassification):

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

        emission = self.predict_proba(X)

        sequence = viterbi_decoding(
            emission, self.transition_,
            initial=self.initial_,
            consecutive=consecutive, constraint=constraint)

        return sequence


class GMMSegmentation(SKLearnMixin):

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

        self.classifier_ = SKLearnGMMSegmentation(
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

        X_iter, y_iter = zip(*list(self.Xy_iter(features_iter,
                                                annotation_iter,
                                                unknown='unique')))

        self.label_converter_ = LabelConverter()
        self.label_converter_.fit(np.hstack(y_iter))

        encoded_y_iter = [self.label_converter_.transform(y) for y in y_iter]
        self.classifier_.fit(X_iter, encoded_y_iter)

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
            for label, duration in min_duration.iteritems():
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



class SKLearnGMMUBMSegmentation(SKLearnGMMUBMClassification):

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
            posteriors = np.vstack([posteriors, unknown_posterior])

        sequence = viterbi_decoding(
            np.log(posteriors), self.transition_,
            initial=self.initial_,
            consecutive=consecutive, constraint=constraint)

        if self.open_set_:
            sequence[sequence == (K - 1)] = -1

        return sequence


class GMMUBMSegmentation(SKLearnMixin):

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

        self.classifier_ = SKLearnGMMUBMSegmentation(
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

        X_iter, y_iter = zip(*list(self.Xy_iter(features_iter,
                                                annotation_iter,
                                                unknown='unique')))

        self.label_converter_ = LabelConverter()
        self.label_converter_.fit(np.hstack(y_iter))

        encoded_y_iter = [self.label_converter_.transform(y) for y in y_iter]
        self.classifier_.fit(X_iter, encoded_y_iter)

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
            for label, duration in min_duration.iteritems():
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

# class ViterbiGMMUBMSegmentation(_ViterbiGMMUBMSegmentation, SKLearnMixin):

#     def __init__(self, n_jobs=1, n_components=1, covariance_type='diag',
#                  random_state=None, thresh=1e-2, min_covar=1e-3,
#                  n_iter=100, n_init=1, params='wmc', init_params='wmc',
#                  adapt_iter=10, adapt_params='m', consecutive=None):

#         super(_ViterbiGMMUBMSegmentation, self).__init__(
#             n_jobs=n_jobs, n_components=n_components, covariance_type=covariance_type,
#             random_state=random_state, thresh=thresh, min_covar=min_covar,
#             n_iter=n_iter, n_init=n_init, params=params, init_params=init_params,
#             adapt_iter=adapt_iter, adapt_params=adapt_params)

#         self.consecutive = consecutive

#     def train(self, annotation_iter, features_iter):

#         # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#         # training data
#         X, y = [], []
#         for _X, _y in self.Xy_iter(annotation_iter, features_iter,
#                                    unknown='unique'):
#             X.append(_X)
#             y.append(_y)
#         X = np.vstack(X)
#         Y = np.hstack(y)  # note: y is used later for transition probabilities
#         # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#         # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#         # GMM/UBM training
#         self._train(X, Y)
#         # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#         # number of classes (+1 for unknown)
#         K = len(self.labels_) + 1

#         # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#         # initial & transition probabilities

#         initial = np.zeros((K, ), dtype=float)
#         transition = np.zeros((K, K), dtype=float)

#         for _y in y:

#             # convert sequence from label to class
#             sequence = self._transform_labels(_y)

#             # increment initial count
#             initial[sequence[0]] += 1

#             # increment transition counts
#             for n, m in pairwise(sequence):
#                 transition[n, m] += 1

#         # log-probabilities
#         self.initial_ = np.log(initial / np.sum(initial))
#         self.transition_ = np.log(transition.T / np.sum(transition, axis=1)).T
#         # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#         return self

#     def apply(self, features, constraint=None):

#         # states and labels
#         labels = self.label_binarizer_.labels
#         K = len(labels)
#         state2label = {stt: lbl for stt, lbl in enumerate(labels)}
#         label2state = {lbl: stt for stt, lbl in enumerate(labels)}

#         # features
#         X = self.X(features)
#         N = X.shape[0]
#         sliding_window = features.sliding_window

#         # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#         # mandatory/forbidden state constraints
#         constraint_ = VITERBI_CONSTRAINT_NONE * np.ones((N, K), dtype=int)
#         if constraint is not None:
#             for segment, _, label, value in constraint.itervalues():
#                 if label not in label2state:
#                     continue
#                 t, dt = sliding_window.segmentToRange(segment)
#                 constraint_[t:t + dt, label2state[label]] = value
#         # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#         # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#         # consecutive state constraints
#         consecutive = np.ones((K, ), dtype=int)
#         if isinstance(self.consecutive, dict):
#             for i, label in enumerate(labels):
#                 duration = self.consecutive.get(label, 0.)
#                 if duration > 0.:
#                     consecutive[i] = sliding_window.durationToSamples(duration)

#         elif self.consecutive is not None:
#             duration = float(self.consecutive)
#             consecutive[:] = sliding_window.durationToSamples(duration)

#         self.consecutive_ = consecutive
#         # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#         sequence = self.predict(X, constraint=constraint_)

#         return self.y2annotation(sequence, sliding_window, labels=state2label)


class ViterbiHMM(object):
    """HMM-based segmentation with Viterbi decoding

    Uses the LBG algorithm to train GMM for each state.

    Parameters
    ----------

    targets : iterable, optional
        List of targets to be recognized.

    min_duration : dict or float, optional
        {target: duration} dictionary providing the minimum duration constraint
        for each label (in seconds). Defaults to no constraint.
        If float, use the same duration for all labels.

    n_components : int, optional
        Number of mixture components in GMM. Defaults to 1.

    covariance_type : string, optional
        String describing the type of covariance parameters to
        use.  Must be one of 'spherical', 'tied', 'diag', 'full'.
        Defaults to 'diag' (the only one supported for now...)

    n_iter : int, optional
        Number of EM iterations to perform during GMM training.
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

    """

    def __init__(self, targets=None, min_duration=0.,
                 n_components=1, covariance_type='diag',
                 random_state=None, thresh=1e-2, min_covar=1e-3, n_iter=10,
                 disturb=0.05, sampling=0,
                 ubm=None, params='m'):
        super(ViterbiHMM, self).__init__()

        self.targets = targets
        self.min_duration = min_duration
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.thresh = thresh
        self.min_covar = min_covar
        self.n_iter = n_iter
        self.disturb = disturb
        self.sampling = sampling

        self.ubm = ubm
        self.params = params

        self._lbg = LBG(n_components=self.n_components,
                        covariance_type=self.covariance_type,
                        random_state=self.random_state, thresh=self.thresh,
                        min_covar=self.min_covar, n_iter=self.n_iter,
                        disturb=self.disturb, sampling=self.sampling)

    def _get_targets(self, annotation_iterator):
        """Get sorted list of targets from training data"""

        targets = set()
        for annotation in annotation_iterator:
            targets.update(annotation.labels())

        return sorted(targets)

    def _get_target_data(self, annotation_iterator, features_iterator, target):
        """Get training data for state `target`"""

        data = np.vstack([
            f.crop(r.label_coverage(target))  # use target regions only
            for r, f in itertools.izip(annotation_iterator, features_iterator)
        ])

        return data

    # specific to HMMClassification
    def _get_initial(self, annotation_iterator):
        """Get initial log-probabilities for all states"""

        N = len(self.targets)
        labelToState = {label: i for i, label in enumerate(self.targets)}

        initial = np.zeros((N, ), dtype=float)

        for annotation in annotation_iterator:
            for _, _, label in annotation.itertracks(label=True):
                initial[labelToState[label]] += 1
                break

        return np.log(initial / np.sum(initial))

    # specific to HMMClassification
    def _get_transition(self, annotation_iterator, features_iterator):
        """Get transition log-probabilities"""

        N = len(self.targets)
        lbl2stt = {label: i for i, label in enumerate(self.targets)}

        transition = np.zeros((N, N), dtype=float)

        for r, f in itertools.izip(annotation_iterator, features_iterator):
            slidingWindow = f.sliding_window
            prev_label = None

            for segment, _, label in r.itertracks(label=True):

                if prev_label is not None:
                    transition[lbl2stt[prev_label], lbl2stt[label]] += 1

                _, n = slidingWindow.segmentToRange(segment)
                transition[lbl2stt[label], lbl2stt[label]] += n - 1

                prev_label = label

        return np.log(1. * transition.T / np.sum(transition, axis=1)).T

    def _adapt_ubm(self, ubm, data):
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

        # copy UBM structure and parameters
        gmm = sklearn.clone(ubm)
        gmm.params = self.params  # only adapt requested parameters
        gmm.n_iter = self.n_iter
        gmm.n_init = 1
        gmm.init_params = ''      # initialize with UBM attributes

        # initialize with UBM attributes
        gmm.weights_ = ubm.weights_
        gmm.means_ = ubm.means_
        gmm.covars_ = ubm.covars_

        # adaptation
        try:
            gmm.fit(data)
        except ValueError, e:
            logging.error(e)

        return gmm


    def _fit_model(self, data):
        """Fit GMM to `data`"""

        gmm = self._lbg.apply(data)
        return gmm

    def fit(self, annotation_iterator, features_iterator):
        """Train HMM

        Parameters
        ----------
        annotation_iterator : iterable

        features_iterator : iterable
        """
        A = list(annotation_iterator)
        F = list(features_iterator)

        # obtain target list from training data
        if not self.targets:
            self.targets = self._get_targets(A)

        # compute initial probability
        self._initial = self._get_initial(A)

        # compute transition probability
        self._transition = self._get_transition(A, F)

        # train target models
        self._model = {}


        for targets in self.ubm:

            # gather data from all targets in group
            data = np.vstack([
                self._get_target_data(A, F, target)
                for target in targets])

            # train UBM
            ubm = self._fit_model(data)

            # adapt UBM to all targets in group
            for target in targets:
                data = self._get_target_data(A, F, target)
                self._model[target] = self._adapt_ubm(ubm, data)

        for target in self.targets:
            if target in self._model:
                continue
            data = self._get_target_data(A, F, target)
            self._model[target] = self._fit_model(data)

    def _sequence_to_annotation(self, sequence, sliding_window):
        """Convert state sequence to labeled annotation"""

        # list of transition indices
        boundaries = list(np.where(np.diff(sequence))[0])
        boundaries = [-1] + boundaries + [len(sequence)]

        # prepare result annotation
        annotation = Annotation()

        for start, end in pairwise(boundaries):

            # infer segment from transition indices and sliding window
            segment = sliding_window.rangeToSegment(start, end - start)

            # get actual label from sequence
            label = self.targets[sequence[start + 1]]

            # save to annotation
            annotation[segment] = label

        return annotation

    def apply(self, features, constraint=None):
        """Apply Viterbi decoding

        Parameters
        ----------
        features : SlidingWindowFeatures
        constraint : optional, Scores
            0 : no constraint
            1 : forbidden label
            2 : mandatory label
        Returns
        -------
        result : Annotation
        """

        X = features.data

        sliding_window = features.sliding_window

        # compute emission probability
        emission = np.vstack([self._model[target].score(X)
                              for target in self.targets]).T

        n_samples, n_states = emission.shape

        # Minimum duration constraints
        consecutive = None
        if self.min_duration:

            # initialize with no constraint
            # (min-duration = 1 sample)
            consecutive = np.ones((len(self.targets)), dtype=int)

            # if min_duration is a number (i.e. not a dict)
            # we make sure to make it a dict with same duration for all targets
            if not isinstance(self.min_duration, dict):
                self.min_duration = {target: float(self.min_duration)
                                     for target in self.targets}

            # deduce minimum number of states from mininimum duration
            for t, target in enumerate(self.targets):
                duration = self.min_duration.get(target, 0.)
                if duration > 0.:
                    consecutive[t] = sliding_window.durationToSamples(duration)

        # State constraints
        constraint_ = VITERBI_CONSTRAINT_NONE * \
            np.ones((n_samples, n_states), dtype=int)

        target2state = {target: k for k, target in enumerate(self.targets)}

        if constraint is not None:

            # remove constraints that do not match one of existing targets
            targets = set(constraint.labels()) & set(self.targets)
            constraint = constraint.subset(targets)

            for segment, _, target, value in constraint.itervalues():

                state = target2state.get(target)

                # get sample range from segment span
                t0, dt = sliding_window.segmentToRange(segment)

                constraint_[t0:t0 + dt, state] = value

        # Viterbi decoding
        sequence = viterbi_decoding(emission, self._transition,
                                    initial=self._initial,
                                    consecutive=consecutive,
                                    constraint=constraint_)

        # convert state sequence back to annotation
        annotation = self._sequence_to_annotation(sequence, sliding_window)

        return annotation
