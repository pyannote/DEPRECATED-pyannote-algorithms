#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2014 CNRS

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

import numpy as np
import itertools
from pyannote.core import Annotation
from pyannote.core.util import pairwise


class SKLearnIOMixin:

    def X(self, features):
        return features.data

    def Xy(self, annotation, features):

        X, y = [], []
        for segment, _, label in annotation.itertracks(label=True):
            _X = features.crop(segment)
            X.append(_X)
            y.extend([label] * _X.shape[0])

        return np.vstack(X), y

    def Xy_iter(self, annotation_iter, features_iter):
        for ann, features in itertools.izip(annotation_iter, features_iter):
            yield self.Xy(ann, features)

    def Xy_stack(self, annotation_iter, features_iter):
        X, y = [], []
        for _X, _y in self.Xy_iter(annotation_iter, features_iter):
            X.append(_X)
            y.extend(_y)
        return np.vstack(X), y

    def y2annotation(self, y, sliding_window, labels=None):

        if labels is None:
            labels = dict()

        annotation = Annotation()

        changes_at = [-1] + list(np.where(np.diff(y))[0]) + [len(y)]

        for t, T in pairwise(changes_at):
            segment = sliding_window.rangeToSegment(t, T - t)
            label = labels.get(y[t + 1], y[t + 1])
            annotation[segment] = label

        return annotation
