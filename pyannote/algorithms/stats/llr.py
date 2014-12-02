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

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import unicode_literals

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.base import BaseEstimator, TransformerMixin


class LLRPassthrough(BaseEstimator, TransformerMixin):

    def fit(self, X, y):
        return self

    def transform(self, X):
        return X


class LLRNaiveBayes(GaussianNB):

    def fit(self, X, y):
        X = X.reshape((-1, 1))
        super(LLRNaiveBayes, self).fit(X, y)
        self.class_prior_[:] = 1. / len(self.class_prior_)  # equal prior
        return self

    def transform(self, X):
        X = X.reshape((-1, 1))
        log_proba = self.predict_log_proba(X)
        return np.diff(log_proba).reshape((-1, ))


class LLRIsotonicRegression(BaseEstimator, TransformerMixin):

    def fit(self, X, y):

        n_samples = X.shape[0]
        _X_, f8 = str('X'), str('f8')
        _y_, i1 = str('y'), str('i1')

        Xy = np.zeros((n_samples, ), dtype=[(_X_, f8), (_y_, i1)])
        Xy[_X_] = X
        Xy[_y_] = y

        sorted_Xy = np.sort(Xy, order=_X_)

        self.regression_ = IsotonicRegression(y_min=1e-4, y_max=1. - 1e-4,
                                              out_of_bounds='clip')

        self.regression_.fit(sorted_Xy[_X_], sorted_Xy[_y_])

        return self

    def transform(self, X):
        p = self.regression_.transform(X)
        return np.log(p) - np.log(1. - p)


def logsumexp(a, b=None, axis=0):
    """{Over|under}flow-robust computation of log(sum(b*exp(a)))

    Parameters
    ----------
    a : numpy array
    b :
    """
    a = np.rollaxis(a, axis)
    vmax = np.nanmax(a, axis=0)
    if b is None:
        out = np.log(np.sum(np.exp(a - vmax), axis=0))
    else:
        b = np.atleast_2d(b).T
        out = np.log(np.sum(b * np.exp(a - vmax), axis=0))
    out += vmax
    return out
