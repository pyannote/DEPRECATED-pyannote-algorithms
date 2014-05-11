#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2012-2014 CNRS (Hervé BREDIN - http://herve.niderb.fr)

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

from hac import HierarchicalAgglomerativeClustering
from model import HACModel
from stop import HACStop
import numpy as np


class SimilarityThresholdStop(HACStop):

    def __init__(self):
        super(SimilarityThresholdStop, self).__init__()

    def initialize(self, threshold=0, **kwargs):
        self.threshold = threshold

    def update(self, merged_clusters, new_cluster, **kwargs):
        pass

    def reached(self, history=None, **kwargs):
        last_iteration = history.iterations[-1]
        return last_iteration.similarity < self.threshold

    def finalize(self, history=None, **kwargs):
        n = len(history.iterations)
        return history[n-1]


class HACLinkageModel(HACModel):

    def __init__(self):
        super(HACLinkageModel, self).__init__()

    def get_model(self, cluster, **kwargs):
        return tuple([cluster])

    def merge_models(self, clusters, models=None, **kwargs):
        if models is None:
            raise ValueError('')

        new_model = []
        for cluster in clusters:
            other_model = models[cluster]
            new_model.extend(other_model)
        return tuple(new_model)

    def is_symmetric(self):
        return False


class CompleteLinkageModel(HACLinkageModel):

    def get_similarity(
        self, cluster1, cluster2, models=None, feature=None, **kwargs
    ):

        if models is None:
            raise ValueError('')

        if feature is None:
            raise ValueError('')

        model1 = models[cluster1]
        model2 = models[cluster2]
        return np.min(
            feature.subset(
                rows=set(model1), columns=set(model2)
            ).df.values
        )


class AverageLinkageModel(HACLinkageModel):

    def get_similarity(
        self, cluster1, cluster2, models=None, feature=None, **kwargs
    ):

        if models is None:
            raise ValueError('')

        if feature is None:
            raise ValueError('')

        model1 = models[cluster1]
        model2 = models[cluster2]
        return np.mean(
            feature.subset(
                rows=set(model1), columns=set(model2)
            ).df.values
        )


class SingleLinkageModel(HACLinkageModel):

    def get_similarity(
        self, cluster1, cluster2, models=None, feature=None, **kwargs
    ):

        if models is None:
            raise ValueError('')

        if feature is None:
            raise ValueError('')

        model1 = models[cluster1]
        model2 = models[cluster2]
        return np.max(
            feature.subset(
                rows=set(model1), columns=set(model2)
            ).df.values
        )


class CompleteLinkageClustering(HierarchicalAgglomerativeClustering):

    def __init__(self, threshold=None):
        model = CompleteLinkageModel()
        stop = SimilarityThresholdStop(threshold=threshold)
        super(CompleteLinkageClustering, self).__init__(model=model, stop=stop)

    def __call__(self, annotation, matrix):
        """
        annotation : Annotation
        matrix : LabelMatrix
            Label similarity matrix
        """
        return super(CompleteLinkageClustering, self).__call__(
            annotation, feature=matrix)


class AverageLinkageClustering(HierarchicalAgglomerativeClustering):

    def __init__(self, threshold=None):
        model = AverageLinkageModel()
        stop = SimilarityThresholdStop(threshold=threshold)
        super(AverageLinkageClustering, self).__init__(model=model, stop=stop)

    def __call__(self, annotation, matrix):
        return super(AverageLinkageClustering, self).__call__(
            annotation, feature=matrix)


class SingleLinkageClustering(HierarchicalAgglomerativeClustering):

    def __init__(self, threshold=None):
        model = SingleLinkageModel()
        stop = SimilarityThresholdStop(threshold=threshold)
        super(SingleLinkageClustering, self).__init__(model=model, stop=stop)

    def __call__(self, annotation, matrix):
        return super(SingleLinkageClustering, self).__call__(
            annotation, feature=matrix)
