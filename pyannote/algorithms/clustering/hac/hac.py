#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2013-2016 CNRS

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

import logging
import numpy as np

from .model import HACModel
from .stop import HACStop
from .constraint import HACConstraint
from .history import HACHistory


class HierarchicalAgglomerativeClustering(object):
    """Generic constrained hierarchical agglomerative clustering

    Parameters
    ----------
    model : HACModel
        Model
    stop : HACStop, optional
        Stopping criterion
    constraint : HACConstraint, optional
        Constraint (not yet implemented)
    logger : optional
    """

    def __init__(self, model, stopping_criterion=None, constraint=None,
                 logger=None):

        super(HierarchicalAgglomerativeClustering, self).__init__()

        assert isinstance(model, HACModel)
        self.model = model

        if stopping_criterion is not None:
            assert isinstance(stopping_criterion, HACStop)
        else:
            stopping_criterion = HACStop()
        self.stopping_criterion = stopping_criterion

        if constraint is not None:
            assert isinstance(constraint, HACConstraint)
        else:
            constraint = HACConstraint()
        self.constraint = constraint

        if logger is None:
            logger = logging.getLogger(__name__)
            logger.addHandler(logging.NullHandler())
        self.logger = logger

    @property
    def current_state(self):
        """Current state"""
        return self._current_state

    @property
    def features(self):
        """Features"""
        return self._features

    @property
    def history(self):
        """History"""
        return self._history

    def _initialize(self, starting_point, features=None):

        """Initialize HAC with one cluster per label

        Parameters
        ----------
        starting_point : Annotation
        features : Feature, optional

        """

        # initialize current status at starting point
        self._current_state = starting_point.copy()

        # store features
        self._features = features

        # initialize history with original annotation
        self._history = HACHistory(self._current_state)

        # initialize constraints
        self.constraint.initialize(parent=self)

        # initialize models
        self.model.initialize(parent=self)

        # initialize stopping criterion
        self.stopping_criterion.initialize(parent=self)

    def _iterate(self):

        while True:

            if len(self.model._models) < 2:
                break

            while True:

                clusters, similarity = self.model.get_candidates(parent=self)
                msg = (
                    "Next merging candidates are "
                    "%s with (similarity = %g)."
                )
                self.logger.debug(msg % (" ".join(str(c) for c in clusters), similarity))

                # if the best we can do is find clusters with -inf similarity,
                # then stop here
                if similarity == -np.inf:
                    break

                # check constraint
                if self.constraint.mergeable(clusters, parent=self):
                    break

                self.model.block(clusters, parent=self)

                msg = "Constraints prevented merging %s."
                self.logger.debug(msg % " ".join(str(c) for c in clusters))

            if similarity == -np.inf:
                msg = "Nothing left to merge."
                self.logger.debug(msg)
                break

            into = clusters[0]

            # == update annotation (rename merged clusters)
            mapping = {cluster: into for cluster in clusters}
            self._current_state = self._current_state.rename_labels(mapping=mapping, copy=True)

            # == update history (keep track of this iteration)
            self._history.add_iteration(
                clusters, similarity, into)

            # == update constraints
            self.constraint.update(clusters, into, parent=self)

            # == update model
            self.model.update(clusters, into, parent=self)

            #  == update stopping criterion
            # (most of the time, this does nothing)
            self.stopping_criterion.update(clusters, into, parent=self)

            # check if stopping criterion is reached
            # and, if so, stop agglomerating...
            if self.stopping_criterion.reached(parent=self):

                msg = "Reached stopping criterion."
                self.logger.debug(msg)
                break

            yield self._current_state

    def __call__(self, starting_point, features=None, callback=None):
        """

        Parameters
        ----------
        starting_point : Annotation
        features : Feature, optional
        callback : function, optional
        """

        # if starting point is empty, there is nothing to do.
        if not starting_point:
            return starting_point

        self._initialize(starting_point, features=features)

        for i, current_state in enumerate(self._iterate()):
            if callback is None:
                continue
            callback(i, self)

        # default behavior is return the result of the penultimate iteration
        # (unless stopping_criterion.finalize is overiden)
        return self.stopping_criterion.finalize(parent=self)
