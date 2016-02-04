#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2013-2014 CNRS (Hervé BREDIN - http://herve.niderb.fr)

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


class HACStop(object):
    """Stopping criterion for hierarchical agglomerative clustering"""

    def __init__(self):
        super(HACStop, self).__init__()

    def initialize(self, parent=None):
        """(Optionally) initialize stopping criterion

        Parameters
        ----------
        parent : HierarchicalAgglomerativeClustering, optional
        """
        pass

    def update(self, merged_clusters, into, parent=None):
        """(Optionally) update stopping criterion internal states after merge

        Parameters
        ----------
        merged_cluster :
        into :
        parent : HierarchicalAgglomerativeClustering, optional
        """
        pass

    def reached(self, parent=None):
        """Check whether the stopping criterion is reached

        Parameters
        ----------
        parent : HierarchicalAgglomerativeClustering, optional

        Returns
        -------
        reached : boolean
            True if the stopping criterion is reached, False otherwise.
        """
        return False

    def finalize(self, parent=None):
        """(Optionally) post-process

        Default behavior is to return result of penultimate iteration when the
        stopping criterion is reached, and the last iteration otherwise.

        Parameters
        ----------
        parent : HierarchicalAgglomerativeClustering, optional

        Returns
        -------
        final : Annotation

        """
        if self.reached(parent=parent):
            final_state = parent.history[-2]
        else:
            final_state = parent.current_state
        return final_state
