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
from xarray import DataArray

class HACModel(object):
    """"""

    def __init__(self, is_symmetric=False):
        super(HACModel, self).__init__()
        self.is_symmetric = is_symmetric

    def __getitem__(self, cluster):
        return self._models[cluster]

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

    def compute_similarity(self, cluster1, cluster2, parent=None):
        raise NotImplementedError('Missing method compute_similarity')

    def compute_similarity_matrix(self, parent=None):
        raise NotImplementedError('')

    def initialize(self, parent=None):

        self._models = {cluster: self.compute_model(cluster, parent=parent)
                        for cluster in parent.current_state.labels()}

        try:
            self._similarity = self.compute_similarity_matrix(parent=None)

        except NotImplementedError as e:

            clusters = list(self._models)
            n_clusters = len(clusters)

            # initialize similarity at -infinity
            self._similarity = DataArray(
                -np.inf * np.ones((n_clusters, n_clusters)),
                [('i', clusters), ('j', clusters)])

            if self.is_symmetric:
                for i, j in combinations(clusters, 2):
                    similarity = self.compute_similarity(i, j, parent=parent)
                    self._similarity.loc[i, j] = similarity
                    self._similarity.loc[j, i] = similarity
            else:
                for i, j in product(clusters, repeat=2):
                    similarity = self.compute_similarity(i, j, parent=parent)
                    self._similarity.loc[i, j] = similarity

    def get_candidates(self, parent=None):
        """
        Returns
        -------
        clusters : tuple
        similarity : float

        """
        _, n_j = self._similarity.shape
        ij = int(self._similarity.argmax())
        i = ij // n_j
        j = ij % n_j

        similarity = self._similarity[i, j].item()
        clusters = [self._similarity.coords['i'][i].item(),
                    self._similarity.coords['j'][j].item()]

        return clusters, similarity

    def block(self, clusters, parent=None):
        for i, j in combinations(clusters, 2):
            self._similarity.loc[i, j] = -np.inf
            self._similarity.loc[j, i] = -np.inf
        return

    def update(self, merged_clusters, into, parent=None):

        self._models[into] = self.compute_merged_model(merged_clusters,
                                                       parent=parent)

        # remove meaning rows and colums
        for cluster in merged_clusters:
            if cluster == into:
                continue
            del self._models[cluster]
            self._similarity = self._similarity.drop(cluster, dim='i')
            self._similarity = self._similarity.drop(cluster, dim='j')

        clusters = list(self._models)

        for j in clusters:
            if j == into:
                continue

            similarity = self.compute_similarity(into, j, parent=parent)
            self._similarity.loc[into, j] = similarity

            if not self.is_symmetric:
                similarity = self.compute_similarity(j, into, parent=parent)

            self._similarity.loc[j, into] = similarity

        return into

    # def get_track_similarity_matrix(self, annotation, feature):
    #
    #     # one cluster per track
    #     tracks = annotation.anonymize_tracks()
    #     clusters = tracks.labels()
    #
    #     clusterMatrix = self.get_similarity_matrix(
    #         clusters, annotation=tracks, feature=feature)
    #
    #     trackMatrix = LabelMatrix()
    #     for s1, t1, c1 in tracks.itertracks(label=True):
    #         for s2, t2, c2 in tracks.itertracks(label=True):
    #             trackMatrix[(s1, t1), (s2, t2)] = clusterMatrix[c1, c2]
    #
    #     return trackMatrix
