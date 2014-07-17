#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2014 CNRS (Hervé BREDIN - http://herve.niderb.fr)

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


def viterbi_decoding(emission, transition, initial):
    """Viterbi decoding

    Parameters
    ----------
    emission : array of shape (n_samples, n_states)
        E[t, i] is the emission log-probabilities of sample t at state i.
    transition : array of shape (n_states, n_states)
        T[i, j] is the transition log-probabilities from state i to state j.
    initial : array of shape (n_states, )
        I[i] is the initial log-probabilities of state i.

    Returns
    -------
    states : array of shape (n_samples, )
        Most probable state sequence
    """

    T, K = emission.shape  # number of observations x number of states

    # ~~ INITIALIZATION ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    V = np.empty((T, K))  # V[t, k] is the probability of the most probable
                          # state sequence for the first t observations that
                          # has k as its final state.
    V[0, :] = emission[0, :] + initial

    P = np.empty((K, T), dtype=int)  # P[k, t] remembers which state was used
                                     # to get from time t-1 to time t at
                                     # state k
    P[:, 0] = range(K)

    # ~~ NESTED-LOOP VERSION ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # for t in range(1, T):
    #     for k in range(K):
    #         tmp = V[t - 1, :] + transition[:, k]
    #         P[k, t] = np.argmax(tmp)
    #         V[t, k] = emission[t, k] + np.max(tmp)

    # ~~ FASTER COLUMN-WISE VERSION ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # for t in range(1, T):
    #     tmp = V[t - 1, :] + transition.T
    #     P[:, t] = np.argmax(tmp, axis=1)
    #     V[t, :] = emission[t, :] + np.max(tmp, axis=1)

    # ~~ FASTER-ER JOINT MAX/ARGMAX VERSION ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    states = range(K)
    for t in range(1, T):
        tmp = V[t - 1, :] + transition.T
        P[:, t] = np.argmax(tmp, axis=1)
        V[t, :] = emission[t, :] + tmp[states, P[:, t]]

    # ~~ BACK-TRACKING ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    X = np.empty((T,), dtype=int)
    X[-1] = np.argmax(V[-1, :])
    for t in range(1, T):
        X[-(t + 1)] = P[X[-t], -t]

    return X


def constrained_viterbi_decoding(emission, transition, initial, constraint):
    """Constrained Viterbi decoding

    Parameters
    ----------
    emission : array of shape (n_samples, n_states)
        E[t, i] is the emission log-probabilities of sample t at state i.
    transition : array of shape (n_states, n_states)
        T[i, j] is the transition log-probabilities from state i to state j.
    initial : array of shape (n_states, )
        I[i] is the initial log-probabilities of state i.
    constraint : array of shape (n_states, )
        C[i] is a the minimum-consecutive-states constraint for state i.
        C[i] = 1 is equivalent to no constraint.

    Returns
    -------
    states : array of shape (n_samples, )
        Most probable state sequence

    Reference
    ---------
    Enrique Garcia-Cejaa and Ramon Brenaa. "Long-term Activities Segmentation
    using Viterbi Algorithm with a k-minimum-consecutive-states Constraint".
    5th International Conference on Ambient Systems, Networks and Technologies

    """

    T, K = emission.shape  # number of observations x number of states

    # ~~ INITIALIZATION ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    V = np.empty((T, K))
    P = np.empty((T, K), dtype=int)
    C = np.empty((T, K), dtype=int)

    for i in range(K):
        V[0, i] = initial[i] + emission[0, i]
        P[0, i] = 0
        C[0, i] = 1

    # ~~ NESTED-LOOP VERSION ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for t in range(1, T):
        for j in range(K):

            if C[t - 1, j] < constraint[j]:
                V[t, j] = V[t - 1, j] + transition[j, j] + emission[t, j]
                P[t, j] = j

            else:

                ok = [(i == j) | ((C[t - 1, i] >= constraint[i]) & (T - t >= constraint[j]))
                      for i in range(K)]
                tmp = [V[t - 1, i] + transition[i, j] if ok[i] else -np.inf
                       for i in range(K)]

                V[t, j] = np.max(tmp) + emission[t, j]
                P[t, j] = np.argmax(tmp)

            C[t, j] = 1 if j != P[t, j] else C[t - 1, j] + 1

    # ~~ BACK-TRACKING ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    X = np.empty((T,), dtype=int)
    X[-1] = np.argmax(V[-1, :])
    for t in range(1, T):
        X[-(t + 1)] = P[-t, X[-t]]

    return X
