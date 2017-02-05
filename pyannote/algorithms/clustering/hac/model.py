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

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import unicode_literals

"""Models for hierarchical agglomerative clustering"""

import numpy as np
from itertools import combinations, product
from sortedcollections import ValueSortedDict


class HACModel(object):
    """

    Parameters
    ----------
    is_symmetric : bool, optional
        Defaults to False


    Attributes
    ----------

    _similarity : ValueSortedDict
    _models : dict

    """

    def __init__(self, is_symmetric=False):
        super(HACModel, self).__init__()
        self.is_symmetric = is_symmetric

    def __getitem__(self, cluster):
        return self._models[cluster]

    # models

    def compute_model(self, cluster, parent=None):
        """Compute model of cluster given current parent state

        Parameters
        ----------
        cluster : hashable
            Cluster identifier
        parent : HierarchicalAgglomerativeClustering, optional

        Returns
        -------
        model : anything
            Cluster model
        """
        raise NotImplementedError('Missing method compute_model')

    def compute_merged_model(self, clusters, parent=None):
        raise NotImplementedError('Missing method compute_merged_model')

    # 1 vs. 1 similarity/distance

    def compute_distance(self, cluster1, cluster2, parent=None):
        raise NotImplementedError('')

    def compute_similarity(self, cluster1, cluster2, parent=None):
        try:
            return -self.compute_distance(cluster1, cluster2, parent=parent)
        except NotImplementedError as e:
            # one must implement one of compute_similarity & compute_distance
            raise NotImplementedError('Missing method compute_similarity')

    # 1 vs. N similarity/distance

    def compute_distances(self, cluster, clusters, dim='i', parent=None):
        raise NotImplementedError('')

    def compute_similarities(self, cluster, clusters, dim='i', parent=None):
        try:
            return -self.compute_distances(cluster, clusters, dim=dim, parent=parent)
        except NotImplementedError as e:
            raise NotImplementedError('')

    # N vs. N similarity/distance

    def compute_distance_matrix(self, parent=None):
        raise NotImplementedError('')

    def compute_similarity_matrix(self, parent=None):
        try:
            return -self.compute_distance_matrix(parent=parent)
        except NotImplementedError as e:
            raise NotImplementedError('')

    def initialize(self, parent=None):

        # one model per cluster in current_state
        self._models = {}
        for cluster in parent.current_state.labels():
            self._models[cluster] = self.compute_model(cluster, parent=parent)

        # list of clusters
        clusters = list(self._models)

        try:
            self._similarity = self.compute_similarity_matrix(parent=parent)

        except NotImplementedError as e:

            n_clusters = len(clusters)

            self._similarity = ValueSortedDict()

            for i, j in combinations(clusters, 2):

                # compute similarity if (and only if) clusters are mergeable
                if not parent.constraint.mergeable([i, j], parent=parent):
                    continue

                similarity = self.compute_similarity(i, j, parent=parent)
                self._similarity[i, j] = similarity

                if not self.is_symmetric:
                    similarity = self.compute_similarity(j, i, parent=parent)
                    self._similarity[j, i] = similarity

    # NOTE - for now this (get_candidates / block) combination assumes
    # that we merge clusters two-by-two...

    def get_candidates(self, parent=None):
        """
        Returns
        -------
        clusters : tuple
        similarity : float

        """
        return self._similarity.peekitem(index=-1)

    def block(self, clusters, parent=None):
        if len(clusters) > 2:
            raise NotImplementedError(
                'Constrained clustering merging 3+ clusters is not supported.'
            )
        i, j = clusters
        self._similarity.pop((i, j), default=None)
        self._similarity.pop((j, i), default=None)

    def update(self, merged_clusters, into, parent=None):

        # compute merged model
        self._models[into] = self.compute_merged_model(merged_clusters,
                                                       parent=parent)

        # remove old models and corresponding similarity
        removed_clusters = list(set(merged_clusters) - set([into]))
        for cluster in removed_clusters:
            del self._models[cluster]

        for i, j in product(removed_clusters, self._models):
            self._similarity.pop((i, j), default=None)
            self._similarity.pop((j, i), default=None)

        # compute new similarities
        # * all at once if model implements compute_similarities
        # * one by one otherwise

        remaining_clusters = list(set(self._models) - set([into]))

        try:

            # all at once (when available)
            similarities = self.compute_similarities(
                into, remaining_clusters, parent=parent)

        except NotImplementedError as e:

            similarities = dict()

            for cluster in remaining_clusters:

                # compute similarity if (and only if) clusters are mergeable
                if not parent.constraint.mergeable([into, cluster], parent=parent):
                    continue

                similarities[into, cluster] = self.compute_similarity(
                    into, cluster, parent=parent)

                if not self.is_symmetric:
                    similarities[cluster, into] = self.compute_similarity(
                        cluster, into, parent=parent)

            self._similarity.update(similarities)
