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


def viterbi_decoding(emission, transition,
                     initial=None, consecutive=None, force=None):
    """Viterbi decoding

    Parameters
    ----------
    emission : array of shape (n_samples, n_states)
        E[t, i] is the emission log-probabilities of sample t at state i.
    transition : array of shape (n_states, n_states)
        T[i, j] is the transition log-probabilities from state i to state j.
    initial : optional, array of shape (n_states, )
        I[i] is the initial log-probabilities of state i.
    consecutive : optional, int or int array of shape (n_states, )
        C[i] is a the minimum-consecutive-states constraint for state i.
        C[i] = 1 is equivalent to no constraint.
    force : optional, array of shape (n_samples, )
        F[t] = i forces sample t to be in state i.
        Use F[t] = -1 for no constraint.

    Returns
    -------
    states : array of shape (n_samples, )
        Most probable state sequence

    References
    ----------
    Enrique Garcia-Cejaa and Ramon Brenaa. "Long-term Activities Segmentation
    using Viterbi Algorithm with a k-minimum-consecutive-states Constraint".
    5th International Conference on Ambient Systems, Networks and Technologies

    """

    T, K = emission.shape  # number of observations x number of states
    states = np.arange(K)  # states 0 to K-1

    # ~~ INITIALIZATION ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    if initial is None:
        initial = np.log(np.ones((K, )) / K)

    if force is None:
        F = -np.ones((T, ), dtype=int)
    else:
        F = np.array(force, dtype=int).reshape((T, ))

    if consecutive is None:
        D = np.ones((K, ), dtype=int)
    elif isinstance(consecutive, int):
        D = consecutive * np.ones((K, ), dtype=int)
    else:
        D = np.array(consecutive, dtype=int).reshape((K, ))

    V = np.empty((T, K))                # V[t, k] is the probability of the
    V[0, :] = emission[0, :] + initial  # most probable state sequence for the
                                        # first t observations that has k as
                                        # its final state.

    # in case time t=0 is forced in a specific state
    if F[0] >= 0:
        # artificially set V[0, k] to -inf if k != F[0]
        V[0, states != F[0]] = -np.inf

    P = np.empty((T, K), dtype=int)  # P[t, k] remembers which state was used
    P[0, :] = states                 # to get from time t-1 to time t at
                                     # state k

    C = np.empty((T, K), dtype=int)  # C[t, k] = n means that the optimal path
    C[0, :] = 1                      # leading to state k at time t has already
                                     # been in state k since time t-n

    # ~~ FORWARD ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for t in range(1, T):

        _transition = np.array(transition)

        if F[t] < 0:
            # zero transition probability for paths
            # that do not match `consecutive` constraint
            for k in states:
                if C[t - 1, k] < D[k]:
                    _transition[k, states != k] = -np.inf

        else:
            # zero transition probability for paths
            # that do not match `force` constraint
            _transition[:, states != F[t]] = -np.inf

        # tmp[k, k'] is the probability of the most probable path
        # leading to state k at time t - 1, plus the probability of
        # transitioning from state k to state k' (at time t)
        tmp = V[t - 1, :] + _transition.T

        P[t, :] = np.argmax(tmp, axis=1)

        # update C[t, :]
        for k in states:
            # optimal path reaching state k at time t
            # actually came from state k at time t-1
            if P[t, k] == k:
                C[t, k] = C[t - 1, k] + 1

            # optimal path reaching state k at time t
            # actually came from state k' != k at time t-1
            else:
                C[t, k] = 1

        V[t, :] = emission[t, :] + tmp[states, P[t, :]]

    # ~~ BACK-TRACKING ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    X = np.empty((T,), dtype=int)
    X[-1] = np.argmax(V[-1, :])
    for t in range(1, T):
        X[-(t + 1)] = P[-t, X[-t]]

    return X


