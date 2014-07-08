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

# Authors
# Hervé BREDIN (http://herve.niderb.fr)

"""Generic dynamic time warping (DTW) algorithm"""

from __future__ import unicode_literals

import numpy as np


class DynamicTimeWarping(object):
    """

    Parameters
    ----------
    vsequence, hsequence : iterable
        (vertical and horizontal) sequences to be aligned.
    vcost : float, optional
        Cost for vertical paths (i, j) --> (i+1, j)
    hcost : float, optional
        Cost for horizontal paths (i, j) --> (i, j+1)
    dcost : float, optional
        Cost for diagonal paths (i, j) --> (i+1, j+1)
    distance_func : func, optional
        Function (vitem, hitem) --> distance between items
    precomputed : np.array, optional
        (H, W)-shaped array with pre-computed distances

    """

    def __init__(self, vsequence, hsequence,
                 distance_func=None, precomputed=None,
                 vcost=1., hcost=1., dcost=1.):

        super(DynamicTimeWarping, self).__init__()

        # sequences to be aligned
        self.vsequence = vsequence
        self.hsequence = hsequence

        # cost for elementary paths
        self.vcost = vcost  # vertical
        self.hcost = hcost  # horizontal
        self.dcost = dcost  # diagonal

        # precomputed distance matrix
        if precomputed is not None:
            self._distance = precomputed

        # on-the-fly distance computation
        elif distance_func is not None:
            H, W = len(vsequence), len(hsequence)
            self._distance = np.empty((H, W))
            self._distance[:] = np.NAN
            self.distance_func = distance_func

        # any other case is not valid
        else:
            raise ValueError('')

    def _get_distance(self, v, h):

        # if distance is not compute already
        # do it once and for all
        if np.isnan(self._distance[v, h]):
            vitem = self.vsequence[v]
            hitem = self.hsequence[h]
            self._distance[v, h] = self.distance_func(vitem, hitem)

        return self._distance[v, h]

    def _get_cost(self):

        height = len(self.vsequence)
        width = len(self.hsequence)

        cost = np.inf * np.ones((height, width))

        # initialize first row and first column
        cost[0, 0] = self._get_distance(0, 0)
        for v in range(1, height):
            cost[v, 0] = cost[v - 1, 0] + self.vcost * self._get_distance(v, 0)
        for h in range(1, width):
            cost[0, h] = cost[0, h - 1] + self.hcost * self._get_distance(0, h)

        for v in range(1, height):
            for h in range(1, width):
                d = self._get_distance(v, h)
                dv = cost[v - 1, h] + self.vcost * d
                dh = cost[v, h - 1] + self.hcost * d
                dd = cost[v - 1, h - 1] + self.dcost * d
                cost[v, h] = min(dv, dh, dd)

        return cost

    def get_path(self):
        """Get lowest cost path

        Returns
        -------
        path : [(0, 0), ..., [(height-1, width-1)]
        """

        # compute cost matrix
        cost = self._get_cost()

        # initialize path at bottom/right
        height, width = len(self.vsequence), len(self.hsequence)
        v, h = height - 1, width - 1
        path = [(v, h)]

        # backtrack from bottom/right to top/left
        while v > 0 or h > 0:

            # backtrack one step
            v, h = min(
                # go left, go up or both?
                [(v - 1, h), (v, h - 1), (v - 1, h - 1)],
                # use cost matrix to choose among allowed paths
                key=lambda (i, j): np.inf if i < 0 or j < 0 else cost[i, j]
            )

            path.append((v, h))

        # reverse path so that it goes from top/left to bottom/right
        return path[::-1]
