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

STARTS_BEFORE = 1
STARTS_AFTER = 2
ENDS_BEFORE = 4
ENDS_AFTER = 8

STARTS_WITH = STARTS_BEFORE | STARTS_AFTER
ENDS_WITH = ENDS_BEFORE | ENDS_AFTER


class DynamicTimeWarping(object):
    """

    Parameters
    ----------
    vsequence, hsequence : iterable
        (vertical and horizontal) sequences to be aligned.
    vcost : float, optional
        Cost for vertical paths (i, j) --> (i+1, j)
        Reducing `vcost` encourages paths where several v-elements are aligned
        with the same h-element (i.e. fine-to-coarse alignments)
    hcost : float, optional
        Cost for horizontal paths (i, j) --> (i, j+1)
        Reducing `hcost` encourages paths where several h-elements are aligned
        with the same v-element (i.e. coarse-to-fine alignments)
    dcost : float, optional
        Cost for diagonal paths (i, j) --> (i+1, j+1)
    distance_func : func, optional
        Function (vitem, hitem) --> distance between items
    distance : np.array, optional
        (H, W)-shaped array with pre-computed distances
    mask_func : func, optional
        Function (vitem, hitem) --> mask
    mask : np.array, optional
        (H, W)-shaped boolean array with pre-computed mask.

    """

    def __init__(self, vsequence, hsequence,
                 vcost=1., hcost=1., dcost=1.,
                 distance_func=None, distance=None,
                 mask_func=None, mask=None,
                 vallow=True, hallow=True):

        super(DynamicTimeWarping, self).__init__()

        # sequences to be aligned
        self.vsequence = vsequence
        self.hsequence = hsequence

        H, W = len(vsequence), len(hsequence)

        # cost for elementary paths
        self.vcost = vcost  # vertical
        self.hcost = hcost  # horizontal
        self.dcost = dcost  # diagonal

        # allow vertical paths
        self.v_ok = vallow
        # allow horizontal paths
        self.h_ok = hallow

        # precomputed distance matrix
        if distance is not None:
            assert distance.shape == (H, W)
            self._distance = distance

        # on-the-fly distance computation
        elif distance_func is not None:
            self._distance = np.empty((H, W))
            self._distance[:] = np.NAN
            self.distance_func = distance_func

        # any other case is not valid
        else:
            raise ValueError('')

        # precomputed mask matrix
        if mask is not None:
            assert mask.shape == (H, W)
            self._mask = mask

        # on-the-fly mask computation
        elif mask_func is not None:
            self._mask = np.empty((H, W))
            self._mask[:] = np.NAN
            self.mask_func = mask_func

        # no mask
        else:
            self._mask = True * np.ones((H, W))

    def _get_distance(self, v, h):
        """Returns distance between elements v and h"""

        # if distance is not computed already
        # do it once and for all
        if np.isnan(self._distance[v, h]):
            vitem = self.vsequence[v]
            hitem = self.hsequence[h]
            self._distance[v, h] = self.distance_func(vitem, hitem)

        return self._distance[v, h]

    def _get_mask(self, v, h):
        """Returns mask at position (v, h)"""

        # if mask is not computed already
        # do it once and for all
        if np.isnan(self._mask[v, h]):
            vitem = self.vsequence[v]
            hitem = self.hsequence[h]
            self._mask[v, h] = self.mask_func(vitem, hitem)

        return self._mask[v, h]

    def _get_cost(self):
        """Compute cost matrix (taking mask into account)"""

        height = len(self.vsequence)
        width = len(self.hsequence)

        # initialize with infinite cost
        cost = np.inf * np.ones((height, width))

        # initialize first row and first column
        cost[0, 0] = self._get_distance(0, 0)

        # update first row of cost matrix
        for v in range(1, height):

            # rest of the first row should remain infinite
            # as soon as one element is masked
            if not self._get_mask(v, 0):
                break

            # update cost based on the previous one on the same row
            cost[v, 0] = cost[v - 1, 0] + self.vcost * self._get_distance(v, 0)

        # update first column of cost matrix
        for h in range(1, width):

            # rest of the first column should remain infinite
            # as soon as one element is masked
            if not self._get_mask(0, h):
                break

            # update cost based on the previous one on the same column
            cost[0, h] = cost[0, h - 1] + self.hcost * self._get_distance(0, h)

        for v in range(1, height):
            for h in range(1, width):

                # no need to update cost if this element is masked
                # (it will remain infinite)
                if not self._get_mask(v, h):
                    continue

                d = self._get_distance(v, h)

                dv = (cost[v - 1, h] + self.vcost * d) if self.v_ok else np.inf
                dh = (cost[v, h - 1] + self.hcost * d) if self.h_ok else np.inf
                dd = cost[v - 1, h - 1] + self.dcost * d

                cost[v, h] = min(dv, dh, dd)

        return cost

    def get_path(self):
        """Get alignment path

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

    def get_alignment(self):
        """Get detailed alignment information

        Returns
        -------
        alignment : dict
            Dictionary indexed by aligned (vitem, hitem) pairs.
            Values are bitwise union (|) of the following flags indicating
            if items start (or end) simultaneously or sequentially:
            STARTS_AFTER, STARTS_BEFORE, ENDS_AFTER, ENDS_BEFORE
            with STARTS_WITH = STARTS_BEFORE | STARTS_AFTER and
            ENDS_WITH = ENDS_AFTER | ENDS_BEFORE and
        """

        path = self.get_path()

        v2h, h2v = {}, {}
        alignment = {}

        for _v, _h in path:
            v = self.vsequence[_v]
            r = self.hsequence[_h]
            v2h[v] = v2h.get(v, []) + [r]
            h2v[r] = h2v.get(r, []) + [v]

        # vertical foward pass
        _h = -1
        for v in self.vsequence:
            h = min(v2h[v])
            if h != _h:
                alignment[v, h] = alignment.get((v, h), 0) | STARTS_AFTER
                _h = h

        # horizontal forward pass
        _v = -1
        for h in self.hsequence:
            v = min(h2v[h])
            if v != _v:
                alignment[v, h] = alignment.get((v, h), 0) | STARTS_BEFORE
                _v = v

        # vertical backward pass
        _h = -1
        for v in reversed(self.vsequence):
            h = max(v2h[v])
            if h != _h:
                alignment[v, h] = alignment.get((v, h), 0) | ENDS_BEFORE
                _h = h

        # horizontal backward pass
        _v = -1
        for h in reversed(self.hsequence):
            v = max(h2v[h])
            if v != _v:
                alignment[v, h] = alignment.get((v, h), 0) | ENDS_AFTER
                _v = v

        return alignment
