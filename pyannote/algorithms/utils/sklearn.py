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

import six.moves
import numpy as np
import itertools


class LabelConverter(object):
    """
    Convert PyAnnote labels to SKLearn indices, and vice-versa
    """

    def fit(self, y):
        """Train label-to-index mapping"""

        self.labels_ = np.unique(y)
        self.open_set_ = None in set(self.labels_)
        if self.open_set_:
            assert self.labels_[0] is None
        return self

    def __iter__(self):
        """Iterate over closed-set labels"""

        if self.open_set_:
            labels = self.labels_[1:]
        else:
            labels = self.labels_
        for label in labels:
            yield label

    def mapping(self):
        """Get label-to-index mapping"""

        if self.open_set_:
            mapping = {label: i - 1
                       for i, label in enumerate(self.labels_)}
        else:
            mapping = {label: i
                       for i, label in enumerate(self.labels_)}
        return mapping

    def inverse_mapping(self):
        """Get index-to-label mapping"""

        if self.open_set_:
            mapping = {i - 1: label
                       for i, label in enumerate(self.labels_)}
        else:
            mapping = {i: label
                       for i, label in enumerate(self.labels_)}
        return mapping

    def transform(self, y):
        """Transform labels into indices"""

        converted_y = np.searchsorted(self.labels_, y)
        if self.open_set_:
            converted_y = converted_y - 1
        return converted_y

    def inverse_transform(self, converted_y):
        """Transform indices into labels"""

        if self.open_set_:
            converted_y = converted_y + 1
        y = self.labels_[converted_y]
        return y

    def fit_transform(self, y):
        return self.fit(y).transform(y)


class SKLearnMixin:
    """
    Extract SKLearn data/labels from PyAnnote features/annotation
    """

    def X(self, features, annotation=None):

        if annotation is None:
            return features.data

        X = []
        for segment, _, label in annotation.itertracks(label=True):

            _X = features.crop(segment)
            X.append(_X)

        return np.vstack(X)

    def X_iter(self, features_iter, annotation_iter=None):

        if annotation_iter is None:
            annotation_iter = itertools.repeat(None)

        for features, annotation in six.moves.zip(features_iter, annotation_iter):
            yield self.X(features, annotation=annotation)

    def X_stack(self, features_iter, annotation_iter=None):

        X = []
        for _X in self.X_iter(features_iter, annotation_iter=annotation_iter):
            X.append(_X)

        return np.vstack(X)

    def Xy(self, features, annotation):

        X, y = [], []
        for segment, _, label in annotation.itertracks(label=True):
            _X = features.crop(segment)
            _y = [label] * _X.shape[0]

            X.append(_X)
            y.extend(_y)

        return np.vstack(X), y

    def Xy_iter(self, features_iter, annotation_iter):

        for features, annotation in six.moves.zip(features_iter, annotation_iter):
            yield self.Xy(features, annotation)

    def Xy_stack(self, features_iter, annotation_iter):
        X, y = [], []
        for _X, _y in self.Xy_iter(features_iter, annotation_iter):
            X.append(_X)
            y.extend(_y)

        return np.vstack(X), y
