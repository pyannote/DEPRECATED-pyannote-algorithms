#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2014 CNRS
# Antoine LAURENT (http://www.antoine-laurent.fr)
# Hervé BREDIN (http://herve.niderb.fr)

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

import itertools
import numpy as np
import networkx as nx

from pyannote.core import T
from dtw import DynamicTimeWarping
from dtw import STARTS_BEFORE, STARTS_WITH, STARTS_AFTER
from dtw import ENDS_BEFORE, ENDS_WITH, ENDS_AFTER

from pyannote.features.text.tfidf import TFIDF


class TranscriptionAlignment(object):
    """Transcriptions alignment algorithm

    This algorithm will temporally align two transcriptions (called `vertical`
    and `horizontal` following standard DTW representation).

    Whenever two drifting times are merged, it will keep the `vertical` one.

    Parameters
    ----------
    vattribute : str
    hattribute : str

    """

    def __init__(self, vattribute, hattribute):

        super(TranscriptionAlignment, self).__init__()

        self.vattribute = vattribute
        self.hattribute = hattribute

    def dtw(self, vindex, hindex, distance):
        return DynamicTimeWarping(vindex, hindex, distance=distance)

    def pairwise_distance(self, vsequence, hsequence):
        """Compute pairwise distance matrix

        Parameters
        ----------
        vsequence : (index, data) iterable
        hsequence : (index, data) iterable

        Returns
        -------
        distance : numpy array
            Shape = len(vsequence) x len(hsequence)
        """

        V, H = len(vsequence), len(hsequence)
        return np.random.randn(V, H)

    def find_best_alignment(self, vsequence, hsequence):
        """Compute sequence alignment

        Alignment values are chosen among possible "bitwise or" combinations of
        the following flags:
        - STARTS_BEFORE: `vindex` starts before `hindex` does,
        - STARTS_AFTER: `hindex` starts before `vindex` does,
        - ENDS_BEFORE: `vindex` ends before `hindex` does,
        - ENDS_AFTER: `hindex` ends before `vindex` does,

        Parameters
        ----------
        vsequence : (vindex, vdata) iterable
        hsequence : (hindex, hdata) iterable

        Returns
        -------
        alignment : dict
            (vindex, hindex)-indexed dictionary describing sequence alignment.
        """

        distance = self.pairwise_distance(vsequence, hsequence)

        vindex, _ = itertools.izip(*vsequence)
        hindex, _ = itertools.izip(*hsequence)

        dtw = self.dtw(vindex, hindex, distance=distance)

        alignment = dtw.get_alignment()

        return alignment

    def merge(self, vtranscription, htranscription, alignment):
        """Merge transcriptions based on their alignment

        Parameters
        ----------
        vtranscription, htranscription : Transcription
        alignment : dict

        Returns
        -------
        merged : Transcription

        """

        # pre-process alignment to prevent this kind of situations:
        # `A` and `a` are merged into `a`
        # `A` and `b` are merged into `b` -- but `A` no longer exists as it is
        #                                    now called `a`!!!

        # starts by creating a graph where an edge between two nodes
        # indicate that the corresponding times will be eventually merged
        # (we also keep track of the origin of the the nodes for later use)
        same = nx.Graph()
        for (v, h), status in alignment.iteritems():
            if (status & STARTS_WITH == STARTS_WITH):
                same.add_edge(('v', v[0]), ('h', h[0]))

            if (status & ENDS_WITH == ENDS_WITH):
                same.add_edge(('v', v[1]), ('h', h[1]))

        # create a mapping
        mapping = {}
        for component in nx.connected_components(same):
            anchored = [t for _, t in component if t.anchored]
            if anchored:
                to = anchored[0]
            else:
                vdrifting = [t for origin, t in component if origin == 'v']
                to = vdrifting[0]

            for t in component:
                mapping[t] = to

        # initialize new "union" transcription
        merged = vtranscription.copy()
        merged.add_edges_from(htranscription.edges(data=True))

        # starts by merging nodes
        for (origin, t), to in mapping.iteritems():
            if t != to:
                # according to Transcription.align documentation,
                # if both t and to are drifting, the resulting graph
                # will only contain `to` (from vertical sequence)
                merged.align(t, to)

        for (v, h), status in alignment.iteritems():

            # connect start times in correct order
            if status & STARTS_WITH != STARTS_WITH:

                if status & STARTS_BEFORE == STARTS_BEFORE:
                    merged.add_edge(mapping.get(('v', v[0]), v[0]),
                                    mapping.get(('h', h[0]), h[0]))

                if status & STARTS_AFTER == STARTS_AFTER:
                    merged.add_edge(mapping.get(('h', h[0]), h[0]),
                                    mapping.get(('v', v[0]), v[0]))

            # connect end times in correct order
            if status & ENDS_WITH != ENDS_WITH:

                if status & ENDS_BEFORE == ENDS_BEFORE:
                    merged.add_edge(mapping.get(('v', v[1]), v[1]),
                                    mapping.get(('h', h[1]), h[1]))

                if status & ENDS_AFTER == ENDS_AFTER:
                    merged.add_edge(mapping.get(('h', h[1]), h[1]),
                                    mapping.get(('v', v[1]), v[1]))

        # remove self loops which might have resulted from various merging
        for t in merged:
            if merged.has_edge(t, t):
                merged.remove_edge(t, t)

        return merged

    def _get_sequence(self, transcription, attribute):
        sequence = []
        for s, e, data in transcription.ordered_edges_iter(data=True):
            if attribute in data:
                sequence.append(((s, e), data[attribute]))
        return sequence

    def __call__(self, vtranscription, htranscription):
        """Align two transcriptions

        Parameters
        ----------
        vtranscription : `Transcription`
        htranscription : `Transcription`

        Returns
        -------
        merged : `Transcription`
        """

        # make sure transcriptions do not share any drifting labels
        # and also keep track of `vertical` mapping so that we can
        # retrieve original `vertical` drifting times at the end
        T.reset()
        vtranscription, vmapping = vtranscription.relabel_drifting_nodes()
        htranscription, _ = htranscription.relabel_drifting_nodes()

        # [(start_time, stop_time), sentence] iterable
        vsequence = self._get_sequence(vtranscription, self.vattribute)
        hsequence = self._get_sequence(htranscription, self.hattribute)

        #
        alignment = self.find_best_alignment(vsequence, hsequence)

        # merge transcriptions based on the optimal alignment
        merged = self.merge(vtranscription, htranscription, alignment)

        # retrieve original `vertical` drifting times
        relabeled, _ = merged.relabel_drifting_nodes(vmapping)

        return relabeled


class TFIDFTranscriptionAlignment(TranscriptionAlignment):

    def __init__(self, vattribute, hattribute, tfidf=None):
        super(TFIDFTranscriptionAlignment, self).__init__(vattribute,
                                                          hattribute)
        if tfidf is None:
            tfidif = TFIDF(binary=True)
        self._tfidf = tfidif

    def pairwise_distance(self, vsequence, hsequence):
        """Compute pairwise distance matrix

        Parameters
        ----------
        vsequence : (index, data) iterable
        hsequence : (index, data) iterable

        Returns
        -------
        distance : numpy array
            Shape = len(vsequence) x len(hsequence)
        """

        _, vsentences = itertools.izip(*vsequence)
        _, hsentences = itertools.izip(*hsequence)
        self._tfidf.fit(vsentences + hsentences)
        V = self._tfidf.transform(vsentences)
        H = self._tfidf.transform(hsentences)
        return 1. - (V * H.T).toarray()


class WordsAlignment(TranscriptionAlignment):

    def pairwise_distance(self, wsequence, ssequence):
        """

        Parameters
        ----------
        wsequence : (index, word) iterable
        ssequence : (index ,sentence) iterable

        Returns
        -------
        distance : (W, S)-shaped array
            where W (resp. S) is the number of words (resp. sentences)
            and distance[w, s] = 0 means sth sentence contains wth word.
        """
        _, words = itertools.izip(*wsequence)
        _, sentences = itertools.izip(*ssequence)
        wordInSentence = np.zeros((len(words), len(sentences)), dtype=int)
        for w, word in enumerate(words):
            for s, sentence in enumerate(sentences):
                wordInSentence[w, s] = word in sentence
        return 1 - wordInSentence

    def dtw(self, windex, sindex, distance):
        return DynamicTimeWarping(windex, sindex, distance=distance,
                                  hallow=False)
