#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2014 CNRS

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

import numpy as np
import itertools

VITERBI_CONSTRAINT_NONE = 0
VITERBI_CONSTRAINT_FORBIDDEN = 1
VITERBI_CONSTRAINT_MANDATORY = 2


def viterbi_decoding(emission, transition,
                     initial=None, consecutive=None, constraint=None):
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
    constraint : optional, array of shape (n_samples, n_states)
        K[t, i] = 1 forbids state i at time t.
        K[t, i] = 2 forces state i at time t.
        Use K[t, i] = 0 for no constraint (default).

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

    # balance initial probabilities when they are not provided
    if initial is None:
        initial = np.log(np.ones((K, )) / K)

    # mandatory[t] = k is state k is mandatory at time t
    # mandatory[t] = -1 means no mandatory state
    mandatory = -np.ones((T,), dtype=int)

    # must_change_state[t] = 1 if one must change state
    # between times t-1 and t according to constraints
    # must_change_state[t] = 0 means there is no need (but it is allowed)
    must_change_state = np.zeros((T, ), dtype=int)

    # deal with constraints, when they are provided
    if constraint is not None:

        # make a copy of emission probabilities before modifying them
        emission = np.array(emission)

        # set emission probability to zero for forbidden states
        emission[
            np.where(constraint == VITERBI_CONSTRAINT_FORBIDDEN)] = -np.inf

        # iterate over mandatory states
        for t, k in itertools.izip(
            *np.where(constraint == VITERBI_CONSTRAINT_MANDATORY)
        ):
            # remember that state k is mandatory at time t
            mandatory[t] = k

            # set emission probability to zero
            # for all states but the mandatory one
            emission[t, states != k] = -np.inf

        # one must change state if it is mandatory at time t-1
        # and then forbidden at time t
        must_change_state[
            np.where(
                (constraint[:-1, :] == VITERBI_CONSTRAINT_MANDATORY) &
                (constraint[1:, :] == VITERBI_CONSTRAINT_FORBIDDEN)
            )[0] + 1
        ] = 1

    # no minimum-consecutive-states constraints
    if consecutive is None:
        D = np.ones((K, ), dtype=int)

    # same value for all states
    elif isinstance(consecutive, int):
        D = consecutive * np.ones((K, ), dtype=int)

    # (potentially) different values per state
    else:
        D = np.array(consecutive, dtype=int).reshape((K, ))

    V = np.empty((T, K))                # V[t, k] is the probability of the
    V[0, :] = emission[0, :] + initial  # most probable state sequence for the
                                        # first t observations that has k as
                                        # its final state.

    P = np.empty((T, K), dtype=int)  # P[t, k] remembers which state was used
    P[0, :] = states                 # to get from time t-1 to time t at
                                     # state k

    C = np.empty((T, K), dtype=int)  # C[t, k] = n means that the optimal path
    C[0, :] = 1                      # leading to state k at time t has already
                                     # been in state k since time t-n

    # ~~ FORWARD ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for t in range(1, T):

        # make a copy of transition probabilities before modifying them
        _transition = np.array(transition)

        # in case there is no state constraint at time t
        # deal with "consecutive states" constraint
        if mandatory[t] < 0 and not must_change_state[t]:

            # zero transition probability for paths
            # that do not match `consecutive` constraint
            for k in states:

                # not yet long enough in state k?
                if C[t - 1, k] < D[k]:
                    # cannot transition from state k to state k' != k
                    _transition[k, states != k] = -np.inf

        # tmp[k, k'] is the probability of the most probable path
        # leading to state k at time t - 1, plus the probability of
        # transitioning from state k to state k' (at time t)
        tmp = (V[t - 1, :] + _transition.T).T

        # optimal path to state k at t comes from state P[t, k] at t - 1
        # (find among all possible states at this time t)
        P[t, :] = np.argmax(tmp, axis=0)

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

        # update V for time t
        V[t, :] = emission[t, :] + tmp[P[t, :], states]

    # ~~ BACK-TRACKING ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    X = np.empty((T,), dtype=int)
    X[-1] = np.argmax(V[-1, :])
    for t in range(1, T):
        X[-(t + 1)] = P[-t, X[-t]]

    return X
