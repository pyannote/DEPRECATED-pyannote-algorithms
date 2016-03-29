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
import six.moves
import numpy as np
from pyannote.core.feature import SlidingWindowFeature
from pyannote.core.scores import Scores
from pyannote.core.annotation import Unknown


class BaseClassification(object):
    """
    """

    def _get_targets(self, annotation_iterator):
        """Get sorted list of targets from training data"""

        targets = set()
        for annotation in annotation_iterator:
            targets.update(annotation.labels())

        return sorted(targets)

    def _get_priors(self, annotation_iterator):

        chart = {}
        for annotation in annotation_iterator:
            for target, duration in annotation.chart():
                if target in self.targets:
                    chart[target] = chart.get(target, 0) + duration
                else:
                    chart[Unknown] = chart.get(Unknown, 0) + duration

        total = np.sum(chart.values())
        return {label: duration / total
                for label, duration in six.iteritems(chart)}

    def _get_all_data(self, annotation_iterator, features_iterator):

        data = np.vstack([
            f.crop(r.get_timeline().coverage())  # use labeled regions only
            for r, f in six.moves.zip(annotation_iterator, features_iterator)
        ])

        return data

    def _get_target_data(self, annotation_iterator, features_iterator, target):

        data = np.vstack([
            f.crop(r.label_coverage(target))  # use target regions only
            for r, f in six.moves.zip(annotation_iterator, features_iterator)
        ])

        return data

    def pre_fit(self, annotation_iterator, features_iterator):
        data = self._get_all_data(annotation_iterator, features_iterator)
        self._background = self._fit_background(data)

    def fit(self, annotation_iterator, features_iterator):

        A = list(annotation_iterator)
        F = list(features_iterator)

        # obtain target list from training data
        if not self.targets:
            self.targets = self._get_targets(A)

        # compute target priors
        self._prior = self._get_priors(A)

        # train target models
        self._model = {}
        for target in self.targets:
            data = self._get_target_data(A, F, target)
            self._model[target] = self._fit_model(data)

    def _aggregate_track_scores(self, data):
        return np.average(data, axis=0)

    def scores(self, segmentation, features):

        # create empty scores to hold all scores
        scores = Scores(uri=segmentation.uri, modality=segmentation.modality)

        # raw features data
        data = features.data

        # target scores
        targets_scores = []
        for target in self.targets:
            target_scores = self._apply_model(self._model[target], data)
            targets_scores.append(target_scores)
        targets_scores = np.vstack(targets_scores).T

        # background scores
        if hasattr(self, '_background'):
            targets_scores = self._apply_background(data, targets_scores)

        # TODO: make it work for any kind of features
        new_features = SlidingWindowFeature(
            targets_scores, features.sliding_window)

        for segment, track in segmentation.itertracks():
            x = self._aggregate_track_scores(new_features.crop(segment))
            for t, target in enumerate(self.targets):
                scores[segment, track, target] = x[t]

        return scores
