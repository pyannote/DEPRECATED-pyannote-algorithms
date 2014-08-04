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

# vertical item starts before/after horizontal item does
STARTS_BEFORE = 1
STARTS_AFTER = 2

# items start simultaneously
STARTS_WITH = STARTS_BEFORE | STARTS_AFTER

# vertical item ends before/after horizontal item does
ENDS_BEFORE = 4
ENDS_AFTER = 8

# items end simultaneously
ENDS_WITH = ENDS_BEFORE | ENDS_AFTER


class DynamicTimeWarping(object):
    """Dynamic time warping

    Implements standard dynamic time warping between two (vertical and
    horizontal) sequences of length V and H respectively.

            * ────────────────>   horizontal
            │ *                    sequence
            │   *
            │     * * *
            │           *
            │             *
            V               *

         vertical
         sequence

    Parameters
    ----------

    vsequence, hsequence : iterable
        Vertical and horizontal sequences to align.

    distance : (V, H)-shaped array, optional
        Pre-computed pairwise distance matrix D where D[v, h] provides the
        distance between the vth item of the vertical sequence and the hth item
        of the horizontal one.

    distance_func : func, optional
        Distance function taking two arguments `vitem` (any item from the
        vertical sequence) and `hitem` (any item from the horizontal sequence)
        and returning their distance as float.
        `distance_func` must be provided in case pre-computed `distance` is not
        available.

    mask : (V, H)-shaped boolean array, optional
        Pre-computed constraint mask M where M[v, h] is True when aligning the
        vth item of the vertical sequence and the hth item of the horizontal
        sequence is permitted, and False when it is not.

    mask_func : func, optional
        Mask function taking two required arguments (`v`, `h`) and two optional
        arguments (`vitem`, `hitem`) and returning True when aligning them is
        permitted and False otherwise. Defaults to all True.

    vcost, hcost, dcost : float, optional
        Extra cost added to each vertical, horizontal and diagonal move.
        For instance, a positive `vcost` will encourage horizontal and diagonal
        paths. All three values default to 0.

    no_vertical, no_horizontal : boolean, optional
        Constrain dynamic time warping to contain only non-vertical (resp.
        non-horizontal) moves. Defaults to False (i.e. no constraint).
    """

    def __init__(self, vsequence, hsequence,
                 distance_func=None, distance=None,
                 vcost=0., hcost=0., dcost=0.,
                 mask_func=None, mask=None,
                 no_vertical=True, no_horizontal=True):

        super(DynamicTimeWarping, self).__init__()

        # vertical and horizontal sequences to be aligned
        self.vsequence = vsequence
        self.hsequence = hsequence

        H, W = len(vsequence), len(hsequence)

        # extra cost for each elementary move
        self.vcost = vcost  # vertical
        self.hcost = hcost  # horizontal
        self.dcost = dcost  # diagonal

        # no vertical move
        self.no_vertical = no_vertical
        # no horizontal move
        self.no_horizontal = no_horizontal

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

        # defaults to no mask
        else:
            self._mask = True * np.ones((H, W))

    def _get_distance(self, v, h):
        """Get distance between vth and hth items"""

        # if distance is not computed already
        # do it once and for all
        if np.isnan(self._distance[v, h]):
            vitem = self.vsequence[v]
            hitem = self.hsequence[h]
            self._distance[v, h] = self.distance_func(vitem, hitem)

        return self._distance[v, h]

    def _get_mask(self, v, h):
        """Get mask at position (v, h)"""

        # if mask is not computed already
        # do it once and for all
        if np.isnan(self._mask[v, h]):
            vitem = self.vsequence[v]
            hitem = self.hsequence[h]
            self._mask[v, h] = self.mask_func(v, vitem, h, hitem)

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
            cost[v, 0] = self.vcost + cost[v - 1, 0] + self._get_distance(v, 0)

        # update first column of cost matrix
        for h in range(1, width):

            # rest of the first column should remain infinite
            # as soon as one element is masked
            if not self._get_mask(0, h):
                break

            # update cost based on the previous one on the same column
            cost[0, h] = self.hcost + cost[0, h - 1] + self._get_distance(0, h)

        for v in range(1, height):
            for h in range(1, width):

                # no need to update cost if this element is masked
                # (it will remain infinite)
                if not self._get_mask(v, h):
                    continue

                dv = self.vcost + cost[v - 1, h]
                dh = self.hcost + cost[v, h - 1]
                dd = self.dcost + cost[v - 1, h - 1]

                cost[v, h] = self._get_distance(v, h) + min(dv, dh, dd)

        return cost

    def get_path(self):
        """Get path with minimum cost from (0, 0) to (V-1, H-1)

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

            candidates = [(v - 1, h - 1)]
            if not self.no_vertical:
                candidates.append((v - 1, h))
            if not self.no_horizontal:
                candidates.append((v, h - 1))

            # backtrack one step
            v, h = min(
                # go left, go up or both?
                candidates,
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

        # dictionary indexed by vsequence items (resp. hsequence items)
        # for each v-item, contains the list of integer index of align h-items
        # and reciprocally
        v2h, h2v = {}, {}
        for _v, _h in path:
            v = self.vsequence[_v]
            r = self.hsequence[_h]
            v2h[v] = v2h.get(v, []) + [_h]
            h2v[r] = h2v.get(r, []) + [_v]

        # see docstring
        alignment = {}

        # vertical foward pass (i.e. in vsequence chronological order)
        _h = None
        for v in self.vsequence:

            # find first h-item aligned with v
            h = self.hsequence[min(v2h[v])]

            # if it's a new one, v starts after h does
            if h != _h:
                alignment[v, h] = alignment.get((v, h), 0) | STARTS_AFTER
                _h = h

        # horizontal forward pass (i.e. in hsequence chronological order)
        _v = None
        for h in self.hsequence:

            # find first v-item aligned with h
            v = self.vsequence[min(h2v[h])]

            # if it is a new one, v starts before h does
            if v != _v:
                alignment[v, h] = alignment.get((v, h), 0) | STARTS_BEFORE
                _v = v

        # vertical backward pass (i.e. in vsequence anti-chronological order)
        _h = None
        for v in reversed(self.vsequence):

            # find last h-item aligned with v
            h = self.hsequence[max(v2h[v])]

            # if it is a new one, v ends before h does
            if h != _h:
                alignment[v, h] = alignment.get((v, h), 0) | ENDS_BEFORE
                _h = h

        # horizontal backward pass (i.e. in hsequence anti-chronological order)
        _v = None
        for h in reversed(self.hsequence):

            # find last v-item aligned with h
            v = self.vsequence[max(h2v[h])]

            # if it is a new one, v ends after h does
            if v != _v:
                alignment[v, h] = alignment.get((v, h), 0) | ENDS_AFTER
                _v = v

        return alignment
