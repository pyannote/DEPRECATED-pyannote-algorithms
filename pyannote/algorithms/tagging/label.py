#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2012-2016 CNRS

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

"""
Label tagging algorithms

They can be used to propagate labels from one ``Annotation`` (called `source`)
to another ``Annotation`` (called `target`).

They act as translation algorithms where each `target` label is either given
a unique `source` label translation or left unchanged.

"""

from .base import BaseTagger
from .mapping import HungarianMapper, ArgMaxMapper, GreedyMapper


class LabelTagger(BaseTagger):
    """Generic label tagging.

    Label tagging algorithms are made of two steps:
        - first, a label mapping algorithm is applied to find the optimal
          mapping between target labels and source labels.
        - then, any target label with a corresponding source label is
          translated using this optimal mapping.

    Parameters
    ----------
    mapper : any BaseMapper subclass
        Mapping algorithm used to find the optimal mapping between source and
        target labels.

    See Also
    --------
    :class:`pyannote.algorithm.mapping.base.BaseMapper`

    """
    def __init__(self, mapper):

        # Label tagging algorithm cannot tag timelines (only annotation).
        super(LabelTagger, self).__init__()

        # keep track of mapper
        self.mapper = mapper

    def _tag_annotation(self, source, target):
        """Perform the actual tagging.

        Parameters
        ----------
        source, target : :class:`pyannote.base.annotation.Annotation`

        Returns
        -------
        tagged : :class:`pyannote.base.annotation.Annotation`
            Tagged target.

        """

        mapping = self.mapper(target, source)
        return target.translate(mapping)


class HungarianTagger(LabelTagger):
    """Label tagging based on the Hungarian label mapping algorithm.

    Relies on the Hungarian mapping algorithm to find the optimal one-to-one
    mapping between `target` and `source` labels.

    Parameters
    ----------
        cost : func
        Cost function for Hungarian mapping algorithms.
        Defaults to cooccurrence matrix (A, B) --> A * B

    Examples
    --------
        >>> tagger = HungarianTagger()
        >>> tagged_target = tagger(source, target)

    See Also
    --------
    :class:`LabelTagger`
    :class:`pyannote.algorithm.mapping.hungarian.HungarianMapper`

    """
    def __init__(self, cost=None):
        mapper = HungarianMapper(cost=cost)
        super(HungarianTagger, self).__init__(mapper)


class GreedyTagger(LabelTagger):
    """Label tagging based on the Greedy label mapping algorithm.

    Relies on the Greedy mapping algorithm to find the optimal one-to-one
    mapping between `target` and `source` labels.

    Parameters
    ----------
        cost : func
        Cost function for Greedy mapping algorithms.
        Defaults to cooccurrence matrix (A, B) --> A * B

    Examples
    --------
        >>> tagger = GreedyTagger()
        >>> tagged_target = tagger(source, target)

    See Also
    --------
    :class:`LabelTagger`
    :class:`pyannote.algorithm.mapping.hungarian.GreedyMapper`

    """
    def __init__(self, cost=None):
        mapper = GreedyMapper(cost=cost)
        super(GreedyTagger, self).__init__(mapper)



class ArgMaxTagger(LabelTagger):
    """Label tagging based on the ArgMax label mapping algorithm.

    Relies on the ArgMax mapping algorithm to find the optimal many-to-one
    mapping between `target` and `source` labels.

    Parameters
    ----------
    cost : func
        See ArgMaxMapper documentation.

    Examples
    --------
        >>> tagger = ArgMaxTagger()
        >>> tagged_target = tagger(source, target)

    See Also
    --------
    :class:`LabelTagger`
    :class:`pyannote.algorithm.mapping.argmax.ArgMaxMapper`

    """
    def __init__(self, cost=None):
        mapper = ArgMaxMapper(cost=cost)
        super(ArgMaxTagger, self).__init__(mapper)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
