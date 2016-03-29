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

    distance_func : func, optional
        Distance function taking two arguments `vitem` (any item from the
        vertical sequence) and `hitem` (any item from the horizontal sequence)
        and returning their distance as float.
        `distance_func` must be provided in case pre-computed `distance` is not
        available.

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

    def __init__(self, distance_func=None, mask_func=None,
                 vcost=0., hcost=0., dcost=0.,
                 no_vertical=False, no_horizontal=False):

        super(DynamicTimeWarping, self).__init__()

        # extra cost for each elementary move
        self.vcost = vcost  # vertical
        self.hcost = hcost  # horizontal
        self.dcost = dcost  # diagonal

        # no vertical move
        self.no_vertical = no_vertical
        # no horizontal move
        self.no_horizontal = no_horizontal

        self.distance_func = distance_func
        self.mask_func = mask_func

    def _get_distance(self, v, h):
        """Get distance between vth and hth items"""

        # if distance is not computed already
        # do it once and for all
        if np.isnan(self._distance[v, h]):
            vitem = self._vsequence[v]
            hitem = self._hsequence[h]
            self._distance[v, h] = self.distance_func(vitem, hitem)

        return self._distance[v, h]

    def _get_mask(self, v, h):
        """Get mask at position (v, h)"""

        # if mask is not computed already
        # do it once and for all
        if np.isnan(self._mask[v, h]):
            vitem = self._vsequence[v]
            hitem = self._hsequence[h]
            self._mask[v, h] = self.mask_func(v, vitem, h, hitem)

        return self._mask[v, h]

    def _initialize(self, vsequence, hsequence, distance, mask):

        self._vsequence = vsequence
        self._hsequence = hsequence

        V = len(self._vsequence)
        H = len(self._hsequence)

        # ~~~ distance matrix ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # precomputed distance matrix
        if distance is not None:
            assert distance.shape == (V, H)
            self._distance = distance

        # on-the-fly distance computation
        elif self.distance_func is not None:
            self._distance = np.empty((V, H))
            self._distance[:] = np.NAN

        # any other case is not valid
        else:
            raise ValueError('')

        # ~~~ mask ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # precomputed mask
        if mask is not None:
            assert mask.shape == (V, H)
            self._mask = mask

        # on-the-fly mask computation
        elif self.mask_func is not None:
            self._mask = np.empty((V, H))
            self._mask[:] = np.NAN

        # defaults to no mask
        else:
            self._mask = True * np.ones((V, H))

    def _compute_cost(self):
        """Compute cost matrix (taking mask into account)"""

        V = len(self._vsequence)
        H = len(self._hsequence)

        # initialize with infinite cost
        cost = np.inf * np.ones((V, H))

        # initialize first row and first column
        cost[0, 0] = self._get_distance(0, 0)

        # update first row of cost matrix
        for v in range(1, V):

            # rest of the first row should remain infinite
            # as soon as one element is masked
            # or if vertical moves are not permitted
            if self.no_vertical or not self._get_mask(v, 0):
                break

            # update cost based on the previous one on the same row
            cost[v, 0] = self.vcost + cost[v - 1, 0] + self._get_distance(v, 0)

        # update first column of cost matrix
        for h in range(1, H):

            # rest of the first column should remain infinite
            # as soon as one element is masked
            # or if horizontal moves are not permitted
            if self.no_horizontal or not self._get_mask(0, h):
                break

            # update cost based on the previous one on the same column
            cost[0, h] = self.hcost + cost[0, h - 1] + self._get_distance(0, h)

        for v in range(1, V):
            for h in range(1, H):

                # no need to update cost if this element is masked
                # (it will remain infinite)
                if not self._get_mask(v, h):
                    continue

                D = []
                dd = self.dcost + cost[v - 1, h - 1]
                D.append(dd)

                if not self.no_vertical:
                    dv = self.vcost + cost[v - 1, h]
                    D.append(dv)

                if not self.no_horizontal:
                    dh = self.hcost + cost[v, h - 1]
                    D.append(dh)

                cost[v, h] = self._get_distance(v, h) + min(D)

        return cost

    def _backtrack(self, cost):

        # initialize path at bottom/right
        V, H = len(self._vsequence), len(self._hsequence)
        v, h = V - 1, H - 1
        path = [(v, h)]

        # backtrack from bottom/right to top/left
        while v > 0 or h > 0:

            # build list of candidate predecessors
            candidates = []
            if v > 0 and h > 0:
                candidates.append((v - 1, h - 1))
            if v > 0 and not self.no_vertical:
                candidates.append((v - 1, h))
            if h > 0 and not self.no_horizontal:
                candidates.append((v, h - 1))

            # backtrack one step
            v, h = min(candidates, key=lambda i_j: cost[i_j[0], i_j[1]])

            path.append((v, h))

        # reverse path so that it goes
        # from top/left to bottom/right
        return path[::-1]

    def __call__(self, vsequence, hsequence, distance=None, mask=None):
        """Get path with minimum cost from (0, 0) to (V-1, H-1)

        Parameters
        ----------

        vsequence, hsequence : iterable
            Vertical and horizontal sequences to align.

        distance : (V, H)-shaped array, optional
            Pre-computed pairwise distance matrix D where D[v, h] provides the
            distance between the vth item of the vertical sequence and the hth
            item of the horizontal one.

        mask : (V, H)-shaped boolean array, optional
            Pre-computed constraint mask M where M[v, h] is True when aligning
            the vth item of the vertical sequence and the hth item of the
            horizontal sequence is permitted, and False when it is not.

        Returns
        -------
        path : [(0, 0), ..., [(height-1, width-1)]
        """

        self._initialize(vsequence, hsequence, distance, mask)
        cost = self._compute_cost()
        path = self._backtrack(cost)
        return path

    def get_alignment(self, vsequence, hsequence, distance=None, mask=None):
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

        path = self.__call__(vsequence, hsequence, distance, mask)

        # dictionary indexed by vsequence items (resp. hsequence items)
        # for each v-item, contains the list of integer index of align h-items
        # and reciprocally
        v2h, h2v = {}, {}
        for _v, _h in path:
            v = self._vsequence[_v]
            r = self._hsequence[_h]
            v2h[v] = v2h.get(v, []) + [_h]
            h2v[r] = h2v.get(r, []) + [_v]

        # see docstring
        alignment = {}

        # vertical foward pass (i.e. in vsequence chronological order)
        _h = None
        for v in self._vsequence:

            # find first h-item aligned with v
            h = self._hsequence[min(v2h[v])]

            # if it's a new one, v starts after h does
            if h != _h:
                alignment[v, h] = alignment.get((v, h), 0) | STARTS_AFTER
                _h = h

        # horizontal forward pass (i.e. in hsequence chronological order)
        _v = None
        for h in self._hsequence:

            # find first v-item aligned with h
            v = self._vsequence[min(h2v[h])]

            # if it is a new one, v starts before h does
            if v != _v:
                alignment[v, h] = alignment.get((v, h), 0) | STARTS_BEFORE
                _v = v

        # vertical backward pass (i.e. in vsequence anti-chronological order)
        _h = None
        for v in reversed(self._vsequence):

            # find last h-item aligned with v
            h = self._hsequence[max(v2h[v])]

            # if it is a new one, v ends before h does
            if h != _h:
                alignment[v, h] = alignment.get((v, h), 0) | ENDS_BEFORE
                _h = h

        # horizontal backward pass (i.e. in hsequence anti-chronological order)
        _v = None
        for h in reversed(self._hsequence):

            # find last v-item aligned with h
            v = self._vsequence[max(h2v[h])]

            # if it is a new one, v ends after h does
            if v != _v:
                alignment[v, h] = alignment.get((v, h), 0) | ENDS_AFTER
                _v = v

        return alignment
