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

from munkres import Munkres
import numpy as np
from pyannote.core.matrix import get_cooccurrence_matrix


class BaseMapper(object):

    def __init__(self, cost=None):
        super(BaseMapper, self).__init__()
        self.cost = get_cooccurrence_matrix if cost is None else cost
        
    def __call__(self, A, B):
        raise NotImplementedError()


class ConservativeDirectMapper(BaseMapper):

    def __call__(self, A, B):

        # Cooccurrence matrix
        matrix = self.cost(A, B)

        # For each row, find the most frequent cooccurring column
        mapping = matrix.argmax(axis=1)

        # and keep this pair only if there is no ambiguity
        mapping = {
            a: b for a, b in mapping.iteritems()
            if np.sum((matrix.subset(rows=set([a])) > 0).df.values) == 1
        }

        return mapping


class ArgMaxMapper(BaseMapper):

    def __call__(self, A, B):

        # Cooccurrence matrix
        matrix = self.cost(A, B)

        # for each row, find the most frequent cooccurring column
        mapping = matrix.argmax(axis=1)

        # and keep this mapping only if they are really cooccurring
        mapping = {a: b for a, b in mapping.iteritems() if matrix[a, b] > 0}

        return mapping

class HungarianMapper(BaseMapper):

    def __init__(self, cost=None):
        super(HungarianMapper, self).__init__(cost=cost)
        self._munkres = Munkres()

    def __call__(self, A, B):

        # Cooccurrence matrix
        matrix = self.cost(A, B)

        # Shape and labels
        nRows, nCols = matrix.shape
        rows = matrix.get_rows()
        cols = matrix.get_columns()

        # Cost matrix
        N = max(nCols, nRows)
        C = np.zeros((N, N))
        C[:nCols, :nRows] = (np.max(matrix.df.values) - matrix.df.values).T

        mapping = {}
        for b, a in self._munkres.compute(C):
            if (b < nCols) and (a < nRows):
                if matrix[rows[a], cols[b]] > 0:
                    mapping[rows[a]] = cols[b]

        return mapping



