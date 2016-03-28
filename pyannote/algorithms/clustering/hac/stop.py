#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2013-2016 CNRS

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


class HACStop(object):
    """Stopping criterion for hierarchical agglomerative clustering"""

    def __init__(self):
        super(HACStop, self).__init__()

    def initialize(self, parent=None):
        """(Optionally) initialize stopping criterion

        Parameters
        ----------
        parent : HierarchicalAgglomerativeClustering, optional
        """
        pass

    def update(self, merged_clusters, into, parent=None):
        """(Optionally) update stopping criterion internal states after merge

        Parameters
        ----------
        merged_cluster :
        into :
        parent : HierarchicalAgglomerativeClustering, optional
        """
        pass

    def reached(self, parent=None):
        """Check whether the stopping criterion is reached

        Parameters
        ----------
        parent : HierarchicalAgglomerativeClustering, optional

        Returns
        -------
        reached : boolean
            True if the stopping criterion is reached, False otherwise.
        """
        return False

    def finalize(self, parent=None):
        """(Optionally) post-process

        Default behavior is to return result of penultimate iteration when the
        stopping criterion is reached, and the last iteration otherwise.

        Parameters
        ----------
        parent : HierarchicalAgglomerativeClustering, optional

        Returns
        -------
        final : Annotation

        """

        # clustering stopped for two possible reasons.
        # either it reached the stopping criterion...
        if self.reached(parent=parent):
            return parent.history.penultimate()

        # ... or there is nothing left to merge
        return parent.history.last()


class SimilarityThreshold(HACStop):

    def __init__(self, threshold=0.0, force=False):
        super(SimilarityThreshold, self).__init__()
        self.threshold = threshold
        self.force = force

    def reached(self, parent=None):

        last_iteration = parent.history.last_iteration()
        if last_iteration is None:
            return False

        _reached = last_iteration.similarity < self.threshold

        if self.force:
            # remember which iteration reached the threshold
            if not hasattr(self, '_reached_at') and _reached:
                self._reached_at = len(parent.history)
            # always return False when forcing complete clustering
            return False

        return _reached

    def finalize(self, parent=None):

        if self.force:
            # clustering was forced to go all the way up to one big cluster
            # therefore we need to reconstruct the state it was when it first
            # reached the stopping criterion
            if hasattr(self, '_reached_at'):
                return parent.history[self._reached_at - 1]
            return parent.history.last()

        # clustering stopped for two possible reasons.
        # either it reached the stopping criterion...
        if self.reached(parent=parent):
            return parent.history.penultimate()

        # ... or there is nothing left to merge
        return parent.history.last()


class DistanceThreshold(SimilarityThreshold):

    def reached(self, parent=None):

        last_iteration = parent.history.last_iteration()
        if last_iteration is None:
            return False

        _reached = last_iteration.similarity < -self.threshold

        if self.force:
            # remember which iteration reached the threshold
            if not hasattr(self, '_reached_at') and _reached:
                self._reached_at = len(parent.history)
            # always return False when forcing complete clustering
            return False

        return _reached


# class InflexionPoint(HACStop):
#
#     def reached(self, parent=None):
#         return False
#
#     def finalize(self, parent=None):
#         y = np.array(i.similarity for i in parent.history.iterations)
#         i = find_inflextion_point(y)
#         return parent.history[i]
