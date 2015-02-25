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
from ..utils.Xy import keepZeroOrOne


class LLRPassthrough(BaseEstimator, TransformerMixin):

    def fit(self, X, y):
        return self

    def transform(self, X):
        return X


class LLRNaiveBayes(GaussianNB):

    def __init__(self, equal_priors=False):
        super(LLRNaiveBayes, self).__init__()
        self.equal_priors = equal_priors

    def fit(self, X, y):

        X, y = keepZeroOrOne(X, y, reshape=((-1, 1)))
        super(LLRNaiveBayes, self).fit(X, y)

        if self.equal_priors:
            self.class_prior_[:] = 1. / len(self.class_prior_)

        return self

    def transform(self, X):
        X = X.reshape((-1, 1))
        log_proba = self.predict_log_proba(X)
        return np.diff(log_proba).reshape((-1, ))


class LLRIsotonicRegression(BaseEstimator, TransformerMixin):

    def __init__(self, equal_priors=False, y_min=1e-4, y_max=1. - 1e-4,
                 plottable=False):
        super(LLRIsotonicRegression, self).__init__()
        self.equal_priors = equal_priors
        self.y_min = y_min
        self.y_max = y_max
        self.plottable = plottable

    def fit(self, X, y):

        X, y = keepZeroOrOne(X, y, reshape=(-1, ))

        if self.plottable:
            self.X_ = X
            self.y_ = y

        if self.equal_priors:

            positive = X[y == 1]
            n_positive = len(positive)
            negative = X[y == 0]
            n_negative = len(negative)

            if n_positive > n_negative:
                # downsample positive examples
                positive = np.random.choice(positive,
                                            size=(n_negative, ),
                                            replace=False)
                n_positive = len(positive)

            else:
                # downsample negative examples
                negative = np.random.choice(negative,
                                            size=(n_positive, ),
                                            replace=False)
                n_negative = len(negative)

            X = np.hstack([negative, positive])
            y = np.hstack([
                np.zeros((n_negative, ), dtype=int),
                np.ones((n_positive, ), dtype=int)
            ])

        n_samples = X.shape[0]

        # hack for numpy
        _X_, f8 = str('X'), str('f8')
        _y_, i1 = str('y'), str('i1')

        Xy = np.zeros((n_samples, ), dtype=[(_X_, f8), (_y_, i1)])
        Xy[_X_] = X
        Xy[_y_] = y

        sorted_Xy = np.sort(Xy, order=_X_)

        self.regression_ = IsotonicRegression(y_min=self.y_min,
                                              y_max=self.y_max,
                                              out_of_bounds='clip')

        self.regression_.fit(sorted_Xy[_X_], sorted_Xy[_y_])

        return self

    def transform(self, X):
        shape = X.shape
        p = self.regression_.transform(X.reshape((-1, )))
        p = p.reshape(shape)
        return np.log(p) - np.log(1. - p)

    def _repr_png_(self):

        from pyannote.core.notebook import plt, _render

        # remember current figure size
        figsize = plt.rcParams['figure.figsize']
        # and update it for segment display
        plt.rcParams['figure.figsize'] = (5, 10)

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1)

        _, bins = np.histogram(self.X_, bins=100)

        mu, sigma = np.mean(self.X_), np.std(self.X_)
        m = mu - 3 * sigma
        M = mu + 3 * sigma

        # m, M = np.min(self.X_), np.max(self.X_)

        positive = self.X_[self.y_ == 1]
        negative = self.X_[self.y_ == 0]
        ax1.hist(positive, bins=bins, alpha=0.5, color='g', normed=True)
        ax1.hist(negative, bins=bins, alpha=0.5, color='r', normed=True)
        ax1.set_xlim(m, M)

        t = np.linspace(m, M, 50)
        ax2.plot(t, self.transform(t))
        ax2.plot([m, M], [0, 0], 'k--')
        ax2.set_xlim(m, M)

        ax3.plot(t, posterior(self.transform(t)))
        ax3.plot([m, M], [0.5, 0.5], 'k--')
        ax3.set_xlim(m, M)
        ax3.set_ylim(-0.1, 1.1)

        data = _render(fig)

        # go back to previous figure size
        plt.rcParams['figure.figsize'] = figsize

        return data


def posterior(llr, prior=None):

    if prior is None:
        prior = 0.5

    priorRatio = (1. - prior) / prior

    return 1 / (1 + priorRatio * np.exp(-llr))


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
