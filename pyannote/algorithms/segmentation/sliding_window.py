#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2012-2016 CNRS

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
import itertools
import numpy as np
import scipy.signal

from pyannote.core import Timeline
from pyannote.core.segment import Segment, SlidingWindow
from ..stats.gaussian import Gaussian
from pyannote.core.util import pairwise


class SlidingWindowsSegmentation(object):
    """

    <---d---><-g-><---d--->
    [   L   ]     [   R   ]
         [   L   ]     [   R   ]
    <-s->

    Parameters
    ----------
    duration : float, optional
        Set left/right window duration. Defaults to 1 second.
    step : float, optional
        Set step duration. Defaults to 100ms
    gap : float, optional
        Set gap duration. Defaults to no gap (i.e. 0 second)
    min_duration : float, optional
        Minimum duration of segments. Defaults to 0 (no minimum).

    """

    def __init__(self, duration=1.0, step=0.1, gap=0.0,
                 threshold=0., min_duration=0., **kwargs):
        super(SlidingWindowsSegmentation, self).__init__()
        self.duration = duration
        self.step = step
        self.gap = gap
        self.threshold = threshold
        self.min_duration = min_duration

        for key, value in six.iteritems(kwargs):
            setattr(self, key, value)

    def diff(self, left, right, feature):
        raise NotImplementedError()

    def iterdiff(self, feature, focus):
        """(middle, difference) generator

        `middle`
        `difference`


        Parameters
        ----------
        feature : SlidingWindowFeature
            Pre-extracted features
        """

        sliding_window = SlidingWindow(
            duration=self.duration,
            step=self.step,
            start=focus.start, end=focus.end)

        for left in sliding_window:

            right = Segment(
                start=left.end + self.gap,
                end=left.end + self.gap + self.duration
            )
            middle = .5 * (left.end + right.start)

            yield middle, self.diff(left, right, feature)

    def apply(self, feature, segmentation=None):

        if segmentation is None:
            focus = feature.getExtent()
            segmentation = Timeline(segments=[focus], uri=None)

        result = Timeline()
        for focus in segmentation:

            x, y = list(zip(*[
                (m, d) for m, d in self.iterdiff(feature, focus)
            ]))
            x = np.array(x)
            y = np.array(y)

            # find local maxima
            order = 1
            if self.min_duration > 0:
                order = int(self.min_duration / self.step)
            maxima = scipy.signal.argrelmax(y, order=order)

            x = x[maxima]
            y = y[maxima]

            # only keep high enough local maxima
            high_maxima = np.where(y > self.threshold)

            # create list of segment boundaries
            # do not forget very first and last boundaries
            boundaries = itertools.chain(
                [focus.start], x[high_maxima], [focus.end]
            )

            # create list of segments from boundaries
            segments = [Segment(*p) for p in pairwise(boundaries)]

            result.update(Timeline(segments=segments))

        return result


class GaussianDivergenceMixin:

    def diff(self, left, right, feature):
        """Compute diagonal gaussian divergence between left and right windows

        Parameters
        ----------
        left, right : Segment
            Left and right window
        feature : Feature

        Returns
        -------
        divergence : float
            Gaussian divergence between left and right windows
        """

        gl = Gaussian(covariance_type='diag')
        Xl = feature.crop(left)
        gl.fit(Xl)

        gr = Gaussian(covariance_type='diag')
        Xr = feature.crop(right)
        gr.fit(Xr)

        try:
            divergence = gl.divergence(gr)
        except:
            divergence = np.NaN

        return divergence


class SegmentationGaussianDivergence(GaussianDivergenceMixin,
                                     SlidingWindowsSegmentation):
    pass
