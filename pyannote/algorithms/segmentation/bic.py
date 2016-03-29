#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2015 CNRS

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

from ..stats.gaussian import RollingGaussian, bayesianInformationCriterion
import numpy as np
from pyannote.core.util import pairwise
from pyannote.core import Timeline, Segment


class SKLearnBICSegmentation(object):

    def __init__(self, penalty_coef=1., covariance_type='full',
                 min_samples=100, precision=10):
        super(SKLearnBICSegmentation, self).__init__()
        self.penalty_coef = penalty_coef
        self.covariance_type = covariance_type
        self.min_samples = min_samples
        self.precision = precision

    def split(self, X, start, end, g1, g2, g):

        g.fit(X, start=start, end=end)

        boundaries = list(range(start + self.min_samples,
                           end - self.min_samples,
                           self.precision))

        bic = np.empty((len(boundaries),))
        for i, boundary in enumerate(boundaries):
            g1.fit(X, start=start, end=boundary)
            g2.fit(X, start=boundary, end=end)
            bic[i] = bayesianInformationCriterion(
                g1, g2, g=g, penalty_coef=self.penalty_coef)

        I = np.argmax(bic)
        BIC = bic[I]
        return boundaries[I] if BIC > 0 else None

    def apply(self, X):

        N = len(X)

        # one gaussian for the whole data
        g = RollingGaussian(covariance_type=self.covariance_type).fit(X)

        # two gaussians (left & right)
        g1 = RollingGaussian(covariance_type=self.covariance_type)
        g2 = RollingGaussian(covariance_type=self.covariance_type)

        start = 0
        end = 3 * self.min_samples

        boundaries = [0, ]
        while end < N:

            boundary = self.split(X, start, end, g1, g2, g)
            if boundary is None:
                end = end + self.min_samples
                continue

            boundaries.append(boundary)
            start = boundary
            end = start + 3 * self.min_samples

        return boundaries + [N - 1]


class BICSegmentation(SKLearnBICSegmentation):
    """
    Parameters
    ----------
    penalty_coef : float, optional
        Set value of penalty coefficient 𝝀. Defaults to 1.
    covariance_type : {'full', 'diag'}, optional
        Set type of covariance matrix. Defaults to 'full'.
    min_duration : int, optional
        Mininum segment duration. Defaults to 1s.
    """

    def __init__(self, penalty_coef=1., covariance_type='full',
                 min_duration=1., precision=0.1):
        super(BICSegmentation, self).__init__()
        self.penalty_coef = penalty_coef
        self.covariance_type = covariance_type
        self.min_duration = min_duration
        self.precision = precision

    def apply(self, features, segmentation=None):
        """
        Parameters
        ----------
        features : Features
        segmentation : Timeline, optional
        """

        if segmentation is None:
            segmentation = Timeline(segments=[features.getExtent()])

        sliding_window = features.sliding_window
        min_samples = sliding_window.durationToSamples(self.min_duration)
        precision = sliding_window.durationToSamples(self.precision)

        segmenter = SKLearnBICSegmentation(
            penalty_coef=self.penalty_coef,
            covariance_type=self.covariance_type,
            min_samples=min_samples,
            precision=precision)

        result = Timeline()

        for long_segment in segmentation:

            X = features.crop(long_segment)
            boundaries = segmenter.apply(X)
            for t, T in pairwise(boundaries):
                segment = sliding_window.rangeToSegment(t, T - t)
                shifted_segment = Segment(long_segment.start + segment.start,
                                          long_segment.start + segment.end)
                result.add(shifted_segment)

        return result
