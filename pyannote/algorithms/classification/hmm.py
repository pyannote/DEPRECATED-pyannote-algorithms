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
from .viterbi import viterbi_decoding
from pyannote.core import Annotation
from pyannote.core.util import pairwise


class ViterbiHMM(object):
    """HMM-based classification with Viterbi decoding

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
                 disturb=0.05, sampling=0):
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
        for target in self.targets:
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
        constraint : optional, Annotation

        Returns
        -------
        result : Annotation
        """

        X = features.data
        sliding_window = features.sliding_window

        # compute emission probability
        emission = np.vstack([self._model[target].score(X)
                              for target in self.targets]).T

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

        # Constraints
        force = None
        if constraint:
            n_samples = emission.shape[0]
            force = -np.ones((n_samples, ), dtype=int)
            target2state = {target: k for k, target in enumerate(self.targets)}
            for segment, _, target in constraint.itertracks(label=True):

                # get sample range from segment span
                t0, dt = sliding_window.segmentToRange(segment)

                # if `target` does not match one of existing targets
                # simply do not take it into account (-1)
                force[t0:t0 + dt] = target2state.get(target, -1)

        # Viterbi decoding
        sequence = viterbi_decoding(emission, self._transition,
                                    initial=self._initial,
                                    consecutive=consecutive,
                                    force=force)

        # convert state sequence to annotation
        annotation = self._sequence_to_annotation(sequence, sliding_window)

        return annotation
