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

from .hac import HierarchicalAgglomerativeClustering
from .hac import HACModel
from .hac.stop import SimilarityThreshold
from pyannote.algorithms.stats.gaussian import Gaussian
import numpy as np
import logging


class BICModel(HACModel):

    def __init__(self, covariance_type='full', penalty_coef=3.5):
        super(BICModel, self).__init__(is_symmetric=True)
        self.covariance_type = covariance_type
        self.penalty_coef = penalty_coef

    def compute_model(self, cluster, parent=None):
        timeline = parent.current_state.label_timeline(cluster)
        data = parent.features.crop(timeline)
        gaussian = Gaussian(covariance_type=self.covariance_type)
        gaussian.fit(data)
        return gaussian

    def compute_merged_model(self, clusters, parent=None):
        gaussian = self[clusters[0]]
        for cluster in clusters[1:]:
            other_gaussian = self[cluster]
            gaussian = gaussian.merge(other_gaussian)
        return gaussian

    def compute_similarity(self, cluster1, cluster2, parent=None):
        gaussian1 = self[cluster1]
        gaussian2 = self[cluster2]
        delta_bic, _ = gaussian1.bic(gaussian2, penalty_coef=self.penalty_coef)
        return -delta_bic


class BICClustering(HierarchicalAgglomerativeClustering):
    """

    Parameters
    ----------
    covariance_type : {'diag', 'full'}, optional
    penalty_coef : float, optional
    """

    def __init__(self, covariance_type='diag', penalty_coef=3.5,
                 logger=None, force=False):

        self.covariance_type = covariance_type
        self.penalty_coef = penalty_coef

        model = BICModel(covariance_type=covariance_type,
                         penalty_coef=penalty_coef)
        stopping_criterion = SimilarityThreshold(threshold=0.0, force=force)

        super(BICClustering, self).__init__(
            model, stopping_criterion=stopping_criterion,
            logger=logger)


class LinearBICClustering(object):
    """Linear BIC clustering

    Parameters
    ----------
    covariance_type : {'diag', 'full'}, optional
        Defaults to 'diag'.
    penalty_coef : float, optional
        Defaults to 1.0
    max_gap : float, optional
        Defaults to infinity (no constraint)

    """
    def __init__(self,
                 covariance_type='diag', penalty_coef=1.0, max_gap=np.inf,
                 logger=None):

        self.covariance_type = covariance_type
        self.penalty_coef = penalty_coef
        self.max_gap = max_gap

        if logger is None:
            logger = logging.getLogger(__name__)
            logger.addHandler(logging.NullHandler())
        self.logger = logger


    def __call__(self, starting_point, features=None):

        current_gaussian = None
        current_label = None
        current_segment = None

        copy = starting_point.copy()

        for segment, track, label in starting_point.itertracks(label=True):

            data = features.crop(segment)
            gaussian = Gaussian(covariance_type=self.covariance_type)
            gaussian.fit(data)

            if current_gaussian is None:
                current_gaussian = gaussian
                current_segment = segment
                current_label = starting_point[segment, track]
                continue

            gap = (current_segment ^ segment).duration
            current_segment = segment

            # stop merging if gap is large
            if gap > self.max_gap:
                current_gaussian = gaussian
                current_label = label
                continue

            delta_bic, merged_gaussian = current_gaussian.bic(
                gaussian, penalty_coef=self.penalty_coef)

            # stop merging if similariy is small
            if delta_bic < 0.0:
                current_gaussian = gaussian
                current_label = label
                continue

            # merge in any other situations
            TEMPLATE = (
                "Merging {cluster1} and {cluster2} with "
                "(BIC = {bic:g})."
            )
            message = TEMPLATE.format(
                cluster1=current_label,
                cluster2=label,
                bic=delta_bic)
            self.logger.debug(message)

            current_gaussian = merged_gaussian
            copy[segment, track] = current_label

        return copy
