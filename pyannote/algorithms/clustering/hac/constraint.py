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

"""Constraints for hierarchical agglomerative clustering"""


from xarray import DataArray
import numpy as np
from pyannote.core import Segment
from scipy.sparse.csgraph import connected_components

class HACConstraint(object):

    def __init__(self):
        super(HACConstraint, self).__init__()

    def initialize(self, parent=None):
        """(Optionally) initialize constraints

        Parameters
        ----------
        parent : HierarchicalAgglomerativeClustering, optional
        """
        pass

    def mergeable(self, clusters, parent=None):
        """Checks whether clusters can be merged

        Parameters
        ----------
        clusters : list
            Candidates for merge.
        parent : HierarchicalAgglomerativeClustering, optional

        Returns
        -------
        mergeable : boolean
            True if clusters can be merged, False otherwise.
        """
        return True

    def update(self, merged_clusters, into, parent=None):
        """(Optionally) update constraints after merge

        Parameters
        ----------
        merged_clusters : list
            List of merged clusters
        into :
            Identifier of resulting cluster
        parent : HierarchicalAgglomerativeClustering, optional
        """
        pass


class _CompoundConstraint(HACConstraint):

    def __init__(self, *constraints):
        super(_CompoundConstraint, self).__init__()
        self.constraints = constraints

    def initialize(self, parent=None):
        for constraint in self.constraints:
            constraint.initialize(parent=parent)

    def update(self, merged_clusters, into, parent=None):
        for constraint in self.constraints:
            constraint.update(merged_clusters, into, parent=parent)


class EveryConstraint(_CompoundConstraint):
    def mergeable(self, clusters, parent=None):
        return all(c.mergeable(clusters, parent=parent)
                   for c in self.constraints)


class AnyConstraint(_CompoundConstraint):
    def mergeable(self, clusters, parent=None):
        return any(c.mergeable(clusters, parent=parent)
                   for c in self.constraints)


class DoNotCooccur(HACConstraint):
    """Do NOT merge co-occurring face tracks"""

    def initialize(self, parent=None):

        current_state = parent.current_state
        clusters = [cluster for cluster in current_state.labels()]
        n_clusters = len(clusters)

        self._cooccur = DataArray(
            np.zeros((n_clusters, n_clusters)),
            [('i', clusters), ('j', clusters)])

        for (segment1, track1), (segment2, track2) in current_state.co_iter(current_state):
            i = current_state[segment1, track1]
            j = current_state[segment2, track2]
            if i == j:
                continue
            self._cooccur.loc[i, j] = 1
            self._cooccur.loc[j, i] = 1

    def mergeable(self, clusters, parent=None):
        clusters = list(clusters)
        return self._cooccur.loc[clusters, clusters].sum().item() == 0.

    def update(self, merged_clusters, new_cluster, parent=None):

        # clusters that will be removed
        _clusters = list(set(merged_clusters) - set([new_cluster]))

        # update coooccurrence matrix
        self._cooccur.loc[new_cluster, :] += self._cooccur.loc[_clusters, :].sum(dim='i')
        self._cooccur.loc[:, new_cluster] += self._cooccur.loc[:, _clusters].sum(dim='j')

        # remove clusters
        self._cooccur = self._cooccur.drop(_clusters, dim='i').drop(_clusters, dim='j')


class CloseInTime(HACConstraint):

    def __init__(self, closer_than=30.0):
        super(CloseInTime, self).__init__()
        self.closer_than = closer_than

    def initialize(self, parent=None):

        current_state = parent.current_state
        extended = current_state.empty()
        for segment, track, cluster in current_state.itertracks(label=True):
            extended_segment = Segment(segment.start - 0.5 * self.closer_than,
                                       segment.end + 0.5 * self.closer_than)
            extended[extended_segment, track] = cluster

        clusters = [cluster for cluster in current_state.labels()]
        n_clusters = len(clusters)

        self._neighbours = DataArray(
            np.zeros((n_clusters, n_clusters)),
            [('i', clusters), ('j', clusters)])

        for (segment1, track1), (segment2, track2) in extended.co_iter(extended):
            i = extended[segment1, track1]
            j = extended[segment2, track2]
            self._neighbours.loc[i, j] += 1
            self._neighbours.loc[j, i] += 1

    def update(self, merged_clusters, new_cluster, parent=None):

        # clusters that will be removed
        _clusters = list(set(merged_clusters) - set([new_cluster]))

        # update coooccurrence matrix
        self._neighbours.loc[new_cluster, :] += self._neighbours.loc[_clusters, :].sum(dim='i')
        self._neighbours.loc[:, new_cluster] += self._neighbours.loc[:, _clusters].sum(dim='j')

        # remove clusters
        self._neighbours = self._neighbours.drop(_clusters, dim='i').drop(_clusters, dim='j')

    def mergeable(self, clusters, parent=None):
        clusters = list(clusters)
        return connected_components(
            self._neighbours.loc[clusters, clusters],
            directed=False, return_labels=False) == 1
