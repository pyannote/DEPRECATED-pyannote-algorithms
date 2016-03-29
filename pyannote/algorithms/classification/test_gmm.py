#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2014-2016 CNRS

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

from .gmm import SKLearnGMMClassification, SKLearnGMMUBMClassification


class Test_GMMClassification:

    def setup(self):

        from sklearn.datasets import load_digits
        dataset = load_digits(n_class=10)

        X = dataset.data
        y = dataset.target

        self.trnX = X[::2]
        self.tstX = X[1::2]
        self.trny = y[::2]
        self.tsty = y[1::2]

    def test_gmm(self):
        gmm = SKLearnGMMClassification(n_components=8)
        gmm.fit(self.trnX, self.trny)
        assert gmm.score(self.tstX, self.tsty) > 0.85

    def test_gmmubm(self):
        gmm = SKLearnGMMUBMClassification(n_components=8)
        gmm.fit(self.trnX, self.trny)
        assert gmm.score(self.tstX, self.tsty) > 0.85

    def test_isotonic(self):
        gmm = SKLearnGMMUBMClassification(n_components=8,
                                          calibration='isotonic')
        gmm.fit(self.trnX, self.trny)
        assert gmm.score(self.tstX, self.tsty) > 0.85
