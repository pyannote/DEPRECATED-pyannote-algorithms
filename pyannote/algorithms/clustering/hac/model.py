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

"""Models for hierarchical agglomerative clustering"""

from pyannote.core.matrix import LabelMatrix


class HACModel(object):
    """"""
    def __init__(self):
        super(HACModel, self).__init__()

    # ==== Clusters Models ===================================================

    def get_model(
        self, cluster,
        annotation=None, models=None, matrix=None, history=None, feature=None
    ):

        """Get model for `cluster`

        Parameters
        ----------
        cluster : hashable
            Cluster unique identifier (typically, annotation label)
        annotation : Annotation, optional
            Annotation at current iteration
        models : dict, optional
            Cluster models at current iteration
        matrix : LabelMatrix, optional
            Cluster similarity matrix at current iteration
        history : HACHistory, optional
            Clustering history up to current iteration
        feature : Feature, optional
            Feature

        Returns
        -------
        model :

        Notes
        -----
        This method must be overriden by inheriting class.
        """

        raise NotImplementedError("Method 'get_model' must be overriden.")

    def get_models(
        self, clusters,
        annotation=None, models=None, matrix=None, history=None, feature=None
    ):

        """Get models for all clusters

        Parameters
        ----------
        clusters : iterable
            Iterable over cluster identifiers
        annotation : Annotation, optional
            Annotation at current iteration
        models : dict, optional
            Cluster models at current iteration
        matrix : LabelMatrix, optional
            Cluster similarity matrix at current iteration
        history : HACHistory, optional
            Clustering history up to current iteration
        feature : Feature, optional
            Feature

        Returns
        -------
        models : dict
            {cluster: model} dictionary for all cluster in `clusters`
        """

        return {
            c: self.get_model(
                c, annotation=annotation, models=models, matrix=matrix,
                history=history, feature=feature
            )
            for c in clusters
        }

    def merge_models(
        self, clusters,
        annotation=None, models=None, matrix=None, history=None, feature=None
    ):

        """Get model resulting from  merging models of all clusters

        Parameters
        ----------
        clusters : iterable
            Iterable over cluster identifiers
        annotation : Annotation, optional
            Annotation at current iteration
        models : dict, optional
            Cluster models at current iteration
        matrix : LabelMatrix, optional
            Cluster similarity matrix at current iteration
        history : HACHistory, optional
            Clustering history up to current iteration
        feature : Feature, optional
            Feature

        Returns
        -------
        model :

        Notes
        -----
        This method must be overriden by inheriting class.
        """

        raise NotImplementedError("Method 'merge_models' must be overriden.")

    # ==== Clusters Similarity ===============================================

    def get_similarity(
        self, cluster1, cluster2,
        annotation=None, models=None, matrix=None, history=None, feature=None
    ):
        """Compute similarity between two clusters

        Parameters
        ----------
        cluster1, cluster2 : hashable
            Cluster unique identifiers (typically, two annotation labels)
        annotation : Annotation, optional
            Annotation at current iteration
        models : dict, optional
            Cluster models at current iteration
        matrix : LabelMatrix, optional
            Cluster similarity matrix at current iteration
        history : HACHistory, optional
            Clustering history up to current iteration
        feature : Feature, optional
            Feature

        Notes
        -----
        This method must be overriden by inheriting class.
        """

        raise NotImplementedError("Method 'get_similarity' must be overriden.")

    def is_symmetric(self):
        """
        Returns
        -------
        symmetric : bool
            True

        Notes
        -----
        This method must be overriden by inheriting class.
        """

        raise NotImplementedError("Method 'is_symmetric' must be overriden.")

    def get_similarity_matrix(
        self, clusters,
        annotation=None, models=None, matrix=None, history=None, feature=None
    ):
        """Compute clusters similarity matrix

        Parameters
        ----------
        clusters : iterable
        annotation : Annotation, optional
            Annotation at current iteration
        models : dict, optional
            Cluster models at current iteration
        matrix : LabelMatrix, optional
            Cluster similarity matrix at current iteration
        history : HACHistory, optional
            Clustering history up to current iteration
        feature : Feature, optional
            Feature

        Returns
        -------
        matrix : LabelMatrix
            Clusters similarity matrix
        """

        if models is None:
            models = {}

        # compute missing models
        models = {
            c: models.get(c, self.get_model(
                c, annotation=annotation, models=models, matrix=matrix,
                history=history, feature=feature))
            for c in clusters
        }

        # cluster similarity matrix
        M = LabelMatrix(
            data=None, dtype=None, rows=clusters, columns=clusters)

        # loop on all pairs of clusters
        for i, cluster1 in enumerate(clusters):
            for j, cluster2 in enumerate(clusters):

                # if similarity is symmetric, no need to compute d(j, i)
                if self.is_symmetric() and j > i:
                    break

                # compute similarity
                M[cluster1, cluster2] = self.get_similarity(
                    cluster1, cluster2, models=models,
                    annotation=annotation, feature=feature)

                # if similarity is symmetric, d(i,j) == d(j, i)
                if self.is_symmetric():
                    M[cluster2, cluster1] = M[cluster1, cluster2]

        return M

    def get_track_similarity_matrix(self, annotation, feature):

        # one cluster per track
        tracks = annotation.anonymize_tracks()
        clusters = tracks.labels()

        clusterMatrix = self.get_similarity_matrix(
            clusters, annotation=tracks, feature=feature)

        trackMatrix = LabelMatrix()
        for s1, t1, c1 in tracks.itertracks(label=True):
            for s2, t2, c2 in tracks.itertracks(label=True):
                trackMatrix[(s1, t1), (s2, t2)] = clusterMatrix[c1, c2]

        return trackMatrix

