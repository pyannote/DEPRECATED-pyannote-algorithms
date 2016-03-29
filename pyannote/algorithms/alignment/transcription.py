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
# Antoine LAURENT - http://www.antoine-laurent.fr
# Hervé BREDIN  - http://herve.niderb.fr


from __future__ import unicode_literals


import six
import six.moves
import numpy as np
import networkx as nx

from pyannote.core import T
from .dtw import DynamicTimeWarping
from .dtw import STARTS_BEFORE, STARTS_WITH, STARTS_AFTER
from .dtw import ENDS_BEFORE, ENDS_WITH, ENDS_AFTER


class OneToAnyMixin:
    """This mixin forbids vertical moves during DTW"""
    @property
    def no_vertical(self):
        return True

    @property
    def no_horizontal(self):
        return False


class AnyToOneMixin:
    """This mixin forbids horizontal moves during DTW"""
    @property
    def no_vertical(self):
        return False

    @property
    def no_horizontal(self):
        return True


class BaseTranscriptionAlignment(object):
    """Base transcriptions alignment algorithm

    This algorithm will temporally align two transcriptions
    (called `vertical` and `horizontal` following standard DTW representation).

            * ────────────────>   horizontal
            │ *
            │   *
            │     * * *
            │           *
            │             *
            V               *

         vertical

    Whenever two drifting times are merged, it will keep the `vertical` one.

    Parameters
    ----------
    vattribute : str
    hattribute : str

    """

    @property
    def no_vertical(self):
        return False

    @property
    def no_horizontal(self):
        return False

    def __init__(self, vcost=0., hcost=0., dcost=0.,):

        super(BaseTranscriptionAlignment, self).__init__()

        self._dtw = DynamicTimeWarping(
            vcost=vcost, hcost=hcost, dcost=dcost,
            no_vertical=self.no_vertical, no_horizontal=self.no_horizontal)

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
        raise NotImplementedError('')

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
        for (v, h), status in six.iteritems(alignment):
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
        for (origin, t), to in six.iteritems(mapping):
            if t != to:
                # according to Transcription.align documentation,
                # if both t and to are drifting, the resulting graph
                # will only contain `to` (from vertical sequence)
                merged.align(t, to)

        for (v, h), status in six.iteritems(alignment):

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

    def _get_sequence(self, transcription, attribute=None):
        """Get raw sequence of attribute values

        Parameters
        ----------
        transcription : `Transcription`
        attribute : str, optional
            When `attribute` is not provided and there are only one attribute
            on the first edge encountered, will automatically select this one
            attribute from all other edges.

        Returns
        -------
        sequence : list
            Chronologically sorted list of ((start_t, end_t), value)
            where `value` is the value of attribute `attribute` on the edge
            between `start_t` and `end_t`.
        """

        sequence = []

        for s, e, data in transcription.ordered_edges_iter(data=True):

            # go to next edge if data is empty
            if not data:
                continue

            # if attribute is not provided and data has only one attribute
            # we choose to use this attribute and make sure the same
            # attribute will be used for the rest of loop.
            # if there is more than one, then raise an error
            if attribute is None:
                if len(data) > 1:
                    msg = 'Which attribute should I use for alignment: {%s}?'
                    raise ValueError(msg % ', '.join(data))
                attribute, item = dict(data).popitem()
                sequence.append(((s, e), item))

            # if attribute is provided (or was chosen automatically above)
            # and data contains it, append the item at the end of the sequence
            elif attribute in data:
                item = data[attribute]
                sequence.append(((s, e), item))

        return sequence

    def __call__(self, vtranscription, htranscription,
                 vattribute=None, hattribute=None):
        """Align two transcriptions

        Parameters
        ----------
        vtranscription, htranscription : `Transcription`
        vattribute, hattribute : str

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

        # compute distance matrix
        vsequence = self._get_sequence(vtranscription, attribute=vattribute)
        hsequence = self._get_sequence(htranscription, attribute=hattribute)
        distance = self.pairwise_distance(vsequence, hsequence)

        # align and merge
        vindex, _ = six.moves.zip(*vsequence)
        hindex, _ = six.moves.zip(*hsequence)
        alignment = self._dtw.get_alignment(vindex, hindex, distance=distance)
        merged = self.merge(vtranscription, htranscription, alignment)

        # retrieve original `vertical` drifting times
        # in case they have not been anchored
        relabeled, _ = merged.relabel_drifting_nodes(vmapping)

        return relabeled


class WordsToSentencesAlignment(AnyToOneMixin, BaseTranscriptionAlignment):

    def pairwise_distance(self, iwords, isentences):
        """

        Parameters
        ----------
        iwords : (index, word) iterable
        isentences : (index ,sentence) iterable

        Returns
        -------
        distance : (W, S)-shaped array
            where W (resp. S) is the number of words (resp. sentences)
            and distance[w, s] = 0 means sth sentence contains wth word.
        """
        _, words = six.moves.zip(*iwords)
        _, sentences = six.moves.zip(*isentences)
        wordInSentence = np.zeros((len(words), len(sentences)), dtype=int)
        for w, word in enumerate(words):
            for s, sentence in enumerate(sentences):
                wordInSentence[w, s] = word in sentence
        return 1 - wordInSentence


class SentencesToWordsAlignment(OneToAnyMixin, WordsToSentencesAlignment):

    def pairwise_distance(self, isentences, iwords):
        D = super(SentencesToWordsAlignment, self).pairwise_distance(
            iwords, isentences)
        return D.T


class TFIDFAlignment(BaseTranscriptionAlignment):
    """

    Parameters
    ----------
    tfidf : `pyannote.features.text.tfidf.TFIDF`
    adapt : boolean, optional
        Whether to adapt `tfidf` to the input sequences (including vocabulary
        and inverse document frequency).
        Default (False) assumes that `tfidf` was trained beforehand.
    """

    def __init__(self, tfidf, adapt=False):
        super(TFIDFAlignment, self).__init__()
        self.tfidf = tfidf
        self.adapt = adapt

    def pairwise_distance(self, vsequence, hsequence):
        """Compute cosine distance in vector space

        Parameters
        ----------
        vsequence : (index, data) iterable
        hsequence : (index, data) iterable

        Returns
        -------
        distance : numpy array
            Shape = len(vsequence) x len(hsequence)
        """

        _, vsentences = six.moves.zip(*vsequence)
        _, hsentences = six.moves.zip(*hsequence)

        if self.adapt:
            self.tfidf.fit(vsentences + hsentences)

        V = self.tfidf.transform(vsentences)
        H = self.tfidf.transform(hsentences)

        return 1. - (V * H.T).toarray()
