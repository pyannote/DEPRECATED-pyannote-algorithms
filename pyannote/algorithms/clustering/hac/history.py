#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2013-2014 CNRS (Hervé BREDIN - http://herve.niderb.fr)

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import unicode_literals

from collections import namedtuple
from networkx import DiGraph, connected_components, topological_sort

class HACIteration(
    namedtuple('HACIteration',
               ['merge', 'similarity', 'into'])
):

    """Iteration of hierarchical agglomerative clustering

    Parameters
    ----------
    merge : iterable
        Unique identifiers of merged clusters
    similarity : float
        Similarity between merged clusters
    into : hashable
        Unique identifier of resulting clusters

    """
    def __new__(cls, merge, similarity, into):
        return super(HACIteration, cls).__new__(
            cls, merge, similarity, into)


class HACHistory(object):
    """History of hierarchical agglomerative clustering

    Parameters
    ----------
    starting_point : Annotation
        Starting point
    iterations : iterable, optional
        HAC iterations in chronological order
    """

    def __init__(self, starting_point, iterations=None):
        super(HACHistory, self).__init__()
        self.starting_point = starting_point.copy()
        if iterations is None:
            self.iterations = []
        else:
            self.iterations = iterations

    def __len__(self):
        return len(self.iterations)

    def add_iteration(self, merge, similarity, into):
        """Add new iteration

        Parameters
        ----------
        merge : iterable
            Unique identifiers of merged clusters
        similarity : float
            Similarity between merged clusters
        into : hashable
            Unique identifier of resulting clusters

        """
        iteration = HACIteration(
            merge=merge,
            similarity=similarity,
            into=into
        )
        self.iterations.append(iteration)

    def last_iteration(self):
        """Return last iteration"""
        if len(self) > 0:
            return self.iterations[-1]
        else:
            return None

    def last(self):
        """Get clustering status after last iteration
        """
        if len(self) > 0:
            return self[-1]
        else:
            return self.starting_point

    def penultimate_iteration(self):
        """Return penultimate iteration"""
        if len(self) > 1:
            return self.iterations[-2]
        else:
            return self.last_iteration()

    def penultimate(self):
        """Get clustering status after penultimate iteration"""
        if len(self) > 1:
            return self[-2]
        else:
            return self.last()

    def __getitem__(self, n):
        """Get clustering status after `n` iterations

        Parameters
        ----------
        n : int
            Number of iterations

        Returns
        -------
        annotation : Annotation
            Clustering status after `n` iterations

        """

        # support for history[-1], history[-2]
        # i = -1 ==> after last iteration
        # i = -2 ==> after penultimate iteration
        # ... etc ...
        if n < 0:
            n = len(self) + 1 + n

        # dendrogram stored as directed graph
        # cluster1 --> cluster2 means cluster1 was merged into cluster2
        g = DiGraph()

        # i = 0 ==> starting point
        # i = 1 ==> after first iteration
        # i = 2 ==> aftr second iterations
        # ... etc ...

        for i, iteration in enumerate(self.iterations):
            if i+1 > n:
                break
            for cluster in iteration.merge:
                if cluster == iteration.into:
                    continue
                g.add_edge(cluster, iteration.into)

        # any cluster is mapped to the last cluster in its topologically
        # ordered connected component
        mapping = {}
        for clusters in connected_components(g.to_undirected()):
            klusters = list(reversed(list(
                topological_sort(g.subgraph(clusters)))))
            for cluster in klusters[1:]:
                mapping[cluster] = klusters[0]

        # actual mapping
        return self.starting_point.rename_labels(mapping=mapping, copy=True)

    def __iter__(self):
        """"""
        annotation = self.starting_point.copy()
        yield annotation
        for iteration in self.iterations:
            translation = {c: iteration.into
                           for c in iteration.merge}
            annotation = annotation % translation
            yield annotation
