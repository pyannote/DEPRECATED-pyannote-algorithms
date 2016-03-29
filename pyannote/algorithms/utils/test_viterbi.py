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

import six.moves
import numpy as np
from .viterbi import viterbi_decoding, \
    VITERBI_CONSTRAINT_NONE, \
    VITERBI_CONSTRAINT_MANDATORY, \
    VITERBI_CONSTRAINT_FORBIDDEN


class TestViterbiDecoding:

    def setup(self):

        from scipy.stats import norm
        from numpy import vstack

        # sample observation from three states with normal density
        F = [norm(0, 0.3), norm(1, 0.45), norm(2, 0.15)]

        # 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2
        # 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 1 2 2 2 2
        self.states = (
            (10 * [0]) + (20 * [1]) + (5 * [2]) + (5 * [1]) +
            (10 * [0]) + (15 * [2]) + (1 * [1]) + (4 * [2])
        )

        obversation = [F[s].rvs() for s in self.states]

        self.emission = vstack([f.logpdf(obversation) for f in F]).T

        self.transition = np.log([[0.5, 0.1, 0.4],
                                  [0.1, 0.8, 0.1],
                                  [0.2, 0.3, 0.5]])

    def test_simple(self):

        decoded = viterbi_decoding(self.emission, self.transition)
        errors = np.sum(decoded != self.states)
        assert float(errors) / len(self.states) < 0.2

    def test_consecutive(self):
        for consecutive in [1, 2, 5, 10, 20, 50]:
            yield self.check_consecutive, consecutive

    def check_consecutive(self, consecutive):

        decoded = viterbi_decoding(self.emission, self.transition,
                                   consecutive=consecutive)

        changeStatesAt = [-1] + list(np.where(np.diff(decoded) != 0)[0])
        lengths = np.diff(changeStatesAt)
        assert np.min(lengths) >= consecutive

    def test_constraint_mandatory(self):
        for ratio in [0.01, 0.05, 0.1, 0.2, 0.5]:
            yield self.check_constraint_mandatory, ratio

    def check_constraint_mandatory(self, ratio):

        T, K = self.emission.shape

        constraint = VITERBI_CONSTRAINT_NONE * np.ones((T, K), dtype=int)

        N = int(ratio * T)
        Ts = np.random.choice(T, size=N, replace=False)
        Ks = np.random.randint(K, size=N)
        for t, k in six.moves.zip(Ts, Ks):
            constraint[t, k] = VITERBI_CONSTRAINT_MANDATORY

        decoded = viterbi_decoding(self.emission, self.transition,
                                   constraint=constraint)

        for i in range(N):
            assert decoded[Ts[i]] == Ks[i]

    def test_constraint_forbidden(self):

        for ratio in [0.01, 0.05, 0.1, 0.2, 0.5]:
            yield self.check_constraint_forbidden, ratio

    def check_constraint_forbidden(self, ratio):

        T, K = self.emission.shape

        constraint = VITERBI_CONSTRAINT_NONE * np.ones((T, K), dtype=int)

        N = int(ratio * T)
        Ts = np.random.choice(T, size=N, replace=False)
        Ks = np.random.randint(K, size=N)
        for t, k in six.moves.zip(Ts, Ks):
            constraint[t, k] = VITERBI_CONSTRAINT_FORBIDDEN

        decoded = viterbi_decoding(self.emission, self.transition,
                                   constraint=constraint)

        for i in range(N):
            assert decoded[Ts[i]] != Ks[i]
