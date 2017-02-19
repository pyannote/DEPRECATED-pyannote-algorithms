#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2012-2017 CNRS

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

import itertools
import numpy as np
from pyannote.algorithms.clustering.hac.hac import HierarchicalAgglomerativeClustering
from pyannote.algorithms.clustering.hac.model import HACModel
from pyannote.algorithms.clustering.hac.stop import SimilarityThreshold


class _LinkageModel(HACModel):

    def __init__(self, is_symmetric=False):
        super(_LinkageModel, self).__init__(
            is_symmetric=is_symmetric)

    def compute_similarity_matrix(self, parent=None):
        return parent.features.copy()

    def compute_model(self, cluster, parent=None):
        return [cluster]

    def compute_merged_model(self, clusters, parent=None):
        merged_model = []
        for cluster in clusters:
            merged_model.extend(self[cluster])
        return list(merged_model)

class CompleteLinkageModel(_LinkageModel):
    def compute_similarity(self, cluster1, cluster2, parent=None):
        return np.min([parent.features[c1, c2]
                      for c1, c2 in itertools.product(self[cluster1], self[cluster2])])

class AverageLinkageModel(_LinkageModel):
    def compute_similarity(self, cluster1, cluster2, parent=None):
        similarities = [
            parent.features[c1, c2]
            for c1, c2 in itertools.product(self[cluster1], self[cluster2])]
        return np.mean(similarities)

class SingleLinkageModel(_LinkageModel):
    def compute_similarity(self, cluster1, cluster2, parent=None):
        return np.max([parent.features[c1, c2]
                      for c1, c2 in itertools.product(self[cluster1], self[cluster2])])

class CompleteLinkageClustering(HierarchicalAgglomerativeClustering):

    def __init__(self, threshold, force=False, logger=None):
        model = CompleteLinkageModel()
        stopping_criterion = SimilarityThreshold(
            threshold=threshold, force=force)
        super(CompleteLinkageClustering, self).__init__(
            model, stopping_criterion=stopping_criterion, logger=logger)

    def __call__(self, starting_point, precomputed, callback=None):
        """
        starting_point : Annotation
        precomputed : ValueSortedDict
            Precomputed cluster similarity matrix
        """
        return super(CompleteLinkageClustering, self).__call__(
            starting_point, features=precomputed, callback=callback)


class AverageLinkageClustering(HierarchicalAgglomerativeClustering):

    def __init__(self, threshold=None, force=False, logger=None):
        model = AverageLinkageModel()
        stopping_criterion = SimilarityThreshold(
            threshold=threshold, force=force)
        super(AverageLinkageClustering, self).__init__(
            model, stopping_criterion=stopping_criterion, logger=logger)

    def __call__(self, starting_point, precomputed, callback=None):
        """
        starting_point : Annotation
        precomputed : ValueSortedDict
            Precomputed cluster similarity matrix
        """
        return super(AverageLinkageClustering, self).__call__(
            starting_point, features=precomputed, callback=callback)


class SingleLinkageClustering(HierarchicalAgglomerativeClustering):

    def __init__(self, threshold=None, force=False, logger=None):
        model = SingleLinkageModel()
        stopping_criterion = SimilarityThreshold(
            threshold=threshold, force=force)
        super(SingleLinkageClustering, self).__init__(
            model, stopping_criterion=stopping_criterion, logger=logger)

    def __call__(self, starting_point, precomputed, callback=None):
        """
        starting_point : Annotation
        precomputed : ValueSortedDict
            Precomputed cluster similarity matrix
        """
        return super(SingleLinkageClustering, self).__call__(
            starting_point, features=precomputed, callback=callback)
