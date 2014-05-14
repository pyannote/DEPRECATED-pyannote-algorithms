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

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import unicode_literals

from pyannote.core import Timeline, Annotation


class BaseTagger(object):
    """Base class for tagging algorithms"""

    def _tag_timeline(self, source, timeline):
        """Must be implemented by inheriting ``Timeline`` tagger.

        Parameters
        ----------
        source : Annotation
            Source annotation whose labels will be propagated
        timeline : Timeline
            Target timeline whose segments will be tagged.

        Returns
        -------
        tagged : Annotation
            Tagged `timeline`

        """
        raise NotImplementedError('Timeline tagging is not supported.')

    def _tag_annotation(self, source, annotation):
        """Must be implemented by inheriting ``Annotation`` tagger.

        Parameters
        ----------
        source : Annotation
            Source annotation whose labels will be propagated
        annotation : Annotation
            Target annotation whose tracks will be tagged.

        Returns
        -------
        tagged : Annotation
            Tagged `annotation`

        """
        raise NotImplementedError('Annotation tagging is not supported.')

    def __call__(self, source, target):
        """Tag `target` based on `source` labels.

        Parameters
        ----------
        source : Annotation
            Source annotation whose labels will be propagated
        target : Timeline or Annotation
            Target timeline (or annotation) whose segment (or tracks) will be
            tagged

        Returns
        -------
        tagged: Annotation
            Tagged target.

        """

        if isinstance(target, Timeline):
            return self._tag_timeline(source, target)

        elif isinstance(target, Annotation):
            return self._tag_annotation(source, target)

if __name__ == "__main__":
    import doctest
    doctest.testmod()
