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

import numpy as np
import sys

from model import HACModel
from stop import HACStop
from constraint import HACConstraint
from history import HACHistory


class HierarchicalAgglomerativeClustering(object):
    """

    Parameters
    ----------
    model : HACModel
        Model
    stop : HACStop, optional
        Stopping criterion
    constraint : HACConstraint, optional
        Constraint (not yet implemented)
    debug : bool, optional

    """

    def __init__(self, model, stop=None, constraint=None, debug=False):

        super(HierarchicalAgglomerativeClustering, self).__init__()

        assert isinstance(model, HACModel)
        self.hacModel = model

        assert isinstance(stop, HACStop)
        self.hacStop = stop

        # assert isinstance(constraint, HACConstraint)
        # self.hacConstraint = constraint

        self.debug = debug

    def initialize(self, annotation, feature=None):

        """Initialize HAC with one cluster per label

        Parameters
        ----------
        annotation : Annotation
        feature : Feature, optional

        """

        # initialize annotation
        self.annotation = annotation.copy()

        # initialize history with original annotation
        self.history = HACHistory(self.annotation)

        # one cluster per label
        clusters = self.annotation.labels()

        # one model per cluster
        self.models = self.hacModel.get_models(
            clusters,
            annotation=self.annotation, feature=feature
        )

        # cluster similarity matrix
        self.matrix = self.hacModel.get_similarity_matrix(
            clusters, models=self.models,
            annotation=self.annotation, feature=feature)

        # make sure diagonals are set to -np.inf
        # -np.inf means "do not merge"
        for c in clusters:
            self.matrix[c, c] = -np.inf

        # TODO: initialize constraints
        # self.hacConstraint.initialize(
        #     annotation=self.annotation, models=self.models,
        #     matrix=self.matrix, history=self.history, feature=feature)

        # initialize stopping criterion
        self.hacStop.initialize(
            annotation=self.annotation, models=self.models,
            matrix=self.matrix, history=self.history, feature=feature)

    def iterate(self, feature=None):

        while True:

            if len(self.models) <= 1:
                break

            # This second loop does not make sense for now.
            # But it will, when we support constrained clustering in the future
            while True:

                # find two most similar clusters
                # TODO: make this block overridable
                #       (e.g. one might want to merge more than 2 clusters at
                #        each iteration)
                cluster1, cluster2 = self.matrix.argmax().popitem()
                similarity = self.matrix[cluster1, cluster2]

                if self.debug:
                    msg = (
                        "DEBUG > Next merging candidates "
                        "are %s and %s (s = %g).\n"
                    )
                    sys.stderr.write(msg % (cluster1, cluster2, similarity))

                # if the best we can do is find clusters with -inf similarity,
                # then stop here
                if similarity == -np.inf:
                    break

                # TODO: constrained clustering
                # if mergeable(cluster1, cluster2)
                #     break
                # self.matrix[cluster1, cluster2] = -np.inf
                # self.matrix[cluster2, cluster1] = -np.inf
                # if self.debug:
                #     msg = "DEBUG > Constraints prevented merging of %s and %s.\n"
                #     sys.stderr.write(msg % (cluster1, cluster2))
                break

            if similarity == -np.inf:
                if self.debug:
                    msg = "DEBUG > Nothing left to merge.\n"
                    sys.stderr.write(msg)
                break
            # == update models

            # (cluster1+cluster2 ==> cluster1)
            self.models[cluster1] = self.hacModel.merge_models(
                [cluster1, cluster2], annotation=self.annotation,
                feature=feature, models=self.models,
                matrix=self.matrix, history=self.history
            )

            # remove (now meaningless) cluster2's model
            del self.models[cluster2]

            # == update annotation (rename cluster2 into cluster1)
            self.annotation = self.annotation % {cluster2: cluster1}

            # == update history (keep track of this iteration)
            self.history.add_iteration(
                [cluster1, cluster2], similarity, cluster1)

            # == update similarity matrix

            # remove (now meaningless) cluster2's row and column
            self.matrix.remove_row(cluster2)
            self.matrix.remove_column(cluster2)

            # update cluster1's row and column
            for cluster in self.models:

                if cluster == cluster1:
                    continue

                # update matrix[cluster1, cluster]
                s = self.hacModel.get_similarity(
                    cluster1, cluster, annotation=self.annotation,
                    models=self.models, matrix=self.matrix,
                    history=self.history, feature=feature
                )
                self.matrix[cluster1, cluster] = s

                # update matrix[cluster, cluster1]
                if not self.hacModel.is_symmetric():
                    s = self.hacModel.get_similarity(
                        cluster, cluster1, annotation=self.annotation,
                        models=self.models, matrix=self.matrix,
                        history=self.history, feature=feature
                    )
                self.matrix[cluster, cluster1] = s

            # TODO:
            # == update constraints

            #  == update stopping criterion
            # (most of the time, this does nothing)
            self.hacStop.update(
                [cluster1, cluster2], cluster1,
                history=self.history, annotation=self.annotation,
                models=self.models, matrix=self.matrix, feature=feature
            )

            # check if stopping criterion is reached
            # and, if so, stop agglomerating...
            if self.hacStop.reached(
                history=self.history,
                annotation=self.annotation, models=self.models,
                matrix=self.matrix, feature=feature
            ):
                if self.debug:
                    msg = "DEBUG > Reached stopping criterion.\n"
                    sys.stderr.write(msg)

                break

            yield self.annotation

    def finalize(self, feature=None):

        self.annotation = self.hacStop.finalize(
            history=self.history, annotation=self.annotation,
            models=self.models, matrix=self.matrix, feature=feature
        )
        return self.annotation

    def __call__(self, annotation, feature=None):

        """

        Parameters
        ----------
        annotation : Annotation
        feature : Feature, optional

        """

        self.initialize(annotation, feature=feature)

        for _ in self.iterate(feature=feature):
            pass

        return self.finalize(feature=feature)
