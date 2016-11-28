#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2012-2016 CNRS

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

import six
from munkres import Munkres
import networkx as nx
import numpy as np


class BaseMapper(object):

    def __init__(self, cost=None):
        super(BaseMapper, self).__init__()
        if cost is None:
            cost = lambda AB: AB[0] * AB[1]
        self.cost = cost

    def __call__(self, A, B):
        raise NotImplementedError()


class ConservativeDirectMapper(BaseMapper):

    def __call__(self, A, B):

        # Cooccurrence matrix
        # for each label in A, find the most cooccurring label in B
        # and keep this pair only if there is no ambiguity
        matrix = self.cost((A, B))
        argmax = matrix.argmax(dim='j').data
        mapping = {a: b for (a, b) in zip(matrix.coords['i'].values,
                                          matrix.coords['j'].values[argmax])
                   if (matrix.loc[a, :] > 0).sum() == 1}

        return mapping


class ArgMaxMapper(BaseMapper):

    def __call__(self, A, B):

        # for each label in A, find the most cooccurring label in B
        matrix = self.cost((A, B))
        argmax = matrix.argmax(dim='j').data
        mapping = {a: b for (a, b) in zip(matrix.coords['i'].values,
                                          matrix.coords['j'].values[argmax])
                   if matrix.loc[a, b] > 0}

        return mapping


class HungarianMapper(BaseMapper):

    def __init__(self, cost=None):
        super(HungarianMapper, self).__init__(cost=cost)
        self._munkres = Munkres()

    def _helper(self, A, B):

        # transpose matrix in case A has more labels than B
        Na = len(A.labels())
        Nb = len(B.labels())
        if Na > Nb:
            return {a: b for (b, a) in six.iteritems(self._helper(B, A))}

        matrix = self.cost((A, B))
        mapping = self._munkres.compute(matrix.max() - matrix)

        return dict(
            (matrix.coords['i'][i].item(), matrix.coords['j'][j].item())
            for i, j in mapping if matrix[i, j] > 0)

    def __call__(self, A, B):

        # build bi-partite cooccurrence graph
        # ------------------------------------

        # labels from A are linked with labels from B
        # if and only if the co-occur
        cooccurrence_graph = nx.Graph()

        # for a_label in A.labels():
        #     a = ('A', a_label)
        #     cooccurrence_graph.add_node(a)
        #
        # for b_label in B.labels():
        #     b = ('B', b_label)
        #     cooccurrence_graph.add_node(b)

        for a_track, b_track in A.co_iter(B):
            a = ('A', A[a_track])
            b = ('B', B[b_track])
            cooccurrence_graph.add_edge(a, b)

        # divide & conquer
        # ------------------

        # split a (potentially large) association problem into smaller ones

        mapping = dict()

        for component in nx.connected_components(cooccurrence_graph):

            # extract smaller problems
            a_labels = [label for (src, label) in component if src == 'A']
            b_labels = [label for (src, label) in component if src == 'B']
            sub_A = A.subset(a_labels)
            sub_B = B.subset(b_labels)

            local_mapping = self._helper(sub_A, sub_B)
            mapping.update(local_mapping)

        return mapping


class GreedyMapper(BaseMapper):

    def __init__(self, cost=None):
        super(GreedyMapper, self).__init__(cost=cost)

    def __call__(self, A, B):

        matrix = self.cost((A, B))
        Na, Nb = matrix.shape
        N = min(Na, Nb)

        mapping = {}

        for i in range(N):

            ab = np.argmax(matrix.data)
            a = ab // (Nb-i)
            b = ab % (Nb-i)

            cost = matrix[a, b].item()

            if cost == 0:
                break

            alabel = matrix.coords['i'][a].item()
            blabel = matrix.coords['j'][b].item()

            mapping[alabel] = blabel

            matrix = matrix.drop([alabel], dim='i').drop([blabel], dim='j')

        return mapping
