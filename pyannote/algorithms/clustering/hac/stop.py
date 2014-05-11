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

class HACStop(object):

    """Stopping criterion for hierarchical agglomerative clustering"""

    def __init__(self):
        super(HACStop, self).__init__()

    def initialize(
        self,
        annotation=None, models=None, matrix=None, history=None, feature=None
    ):

        """
        Parameters
        ----------
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

        """

        raise NotImplementedError("Method 'initialize' must be overriden.")

    def update(
        self, merged_clusters, new_cluster,
        annotation=None, models=None, matrix=None, history=None, feature=None
    ):

        """

        Parameters
        ----------
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
        """

        raise NotImplementedError("Method 'update' must be overriden.")

    def reached(
        self,
        annotation=None, models=None, matrix=None, history=None, feature=None
    ):

        """

        Parameters
        ----------
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

        """

        raise NotImplementedError("Method 'reached' must be overriden.")

    def finalize(
        self,
        annotation=None, models=None, matrix=None, history=None, feature=None
    ):
        """

        Parameters
        ----------
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
        final : Annotation
            Annotation when stop criterion is reached

        """

        raise NotImplementedError("Method 'finalize' must be overriden.")
