#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2012-2015 CNRS

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

from hac import HierarchicalAgglomerativeClustering
from hac import HACModel
from hac import HACStop
from pyannote.algorithms.stats.gaussian import Gaussian


class BICModel(HACModel):
    """
    """

    def __init__(self, covariance_type='full', penalty_coef=3.5):
        super(BICModel, self).__init__()
        self.covariance_type = covariance_type
        self.penalty_coef = penalty_coef

    def get_model(
        self, cluster, annotation=None, feature=None, **kwargs
    ):

        timeline = annotation.label_timeline(cluster)
        data = feature.crop(timeline)
        gaussian = Gaussian(covariance_type=self.covariance_type)
        gaussian.fit(data)
        return gaussian

    def merge_models(
        self, clusters, models=None, annotation=None, feature=None, **kwargs
    ):
        if models is None:
            models = {}

        gaussians = {
            c: models.get(c, self.get_model(
                c, annotation=annotation, feature=feature))
            for c in clusters
        }

        gaussian = gaussians.popitem()[1]
        while gaussians:
            other_gaussian = gaussians.popitem()[1]
            gaussian = gaussian.merge(other_gaussian)

        return gaussian

    def get_similarity(
        self, cluster1, cluster2,
        annotation=None, models=None, matrix=None, history=None, feature=None
    ):

        if models is None:
            models = {}

        if cluster1 in models:
            gaussian1 = models[cluster1]
        else:
            gaussian1 = self.get_model(
                cluster1, annotation=annotation, feature=feature
            )

        if cluster2 in models:
            gaussian2 = models[cluster2]
        else:
            gaussian2 = self.get_model(
                cluster2, annotation=annotation, feature=feature
            )

        dbic, _ = gaussian1.bic(gaussian2, penalty_coef=self.penalty_coef)
        return -dbic

    def is_symmetric(self):
        return True


class BICStop(HACStop):

    def __init__(self):
        super(BICStop, self).__init__()

    def initialize(self, **kwargs):
        pass

    def update(self, merged_clusters, new_cluster, **kwargs):
        pass

    def reached(self, history=None, **kwargs):
        last_iteration = history.iterations[-1]
        return last_iteration.similarity < 0

    def finalize(self, history=None, **kwargs):
        n = len(history.iterations)
        return history[n - 1]


class BICClustering(HierarchicalAgglomerativeClustering):

    def __init__(self, covariance_type='full', penalty_coef=3.5, **kwargs):

        stop = BICStop()
        model = BICModel(
            covariance_type=covariance_type, penalty_coef=penalty_coef
        )
        super(BICClustering, self).__init__(model=model, stop=stop, **kwargs)


# class ContiguousConstraint(HACConstraint):

#     def __init__(self, gap=0.0):
#         super(ContiguousConstraint, self).__init__()
#         self.gap = gap

#     def initialize(
#         self,
#         annotation=None, models=None, matrix=None, history=None, feature=None
#     ):
#         pass

#     def update(
#         self, merged_clusters, new_cluster,
#         annotation=None, models=None, matrix=None, history=None, feature=None
#     ):
#         pass

#     def met(
#         self, clusters,
#         annotation=None, models=None, matrix=None, history=None, feature=None
#     ):
#         pass


# class BICLinearClustering(HierarchicalAgglomerativeClustering):

#     def __init__(
#         self, covariance_type='diag', penalty_coef=1.0, gap=0.0
#     ):

#         stop = BICStop()
#         model = BICModel(
#             covariance_type=covariance_type, penalty_coef=penalty_coef
#         )

#         constraint = ContiguousConstraint(gap=gap)

#         super(BICLinearClustering, self).__init__(
#             model=model, stop=stop, constraint=constraint)
