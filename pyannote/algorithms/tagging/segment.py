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

"""Segment tagging algorithms"""

from .base import BaseTagger


class DirectTagger(BaseTagger):
    """
    Direct segment tagger

    **Timeline tagging.**
    Each segment in target timeline is tagged with all intersecting labels.

    """

    def _tag_timeline(self, source, timeline):
        """Timeline tagging

        Each segment in target `timeline` is tagged with all intersecting
        labels.

        Parameters
        ----------
        source : Annotation
            Source annotation whose labels will be propagated
        timeline : Timeline
            Target timeline whose segments will be tagged.

        Returns
        -------
        tagged : Annotation
            Tagged `timeline`, one track per intersecting label

        """

        # initialize tagged timeline as an empty copy of source
        # (i.e. same uri and same modality)
        tagged = source.empty()

        # loop on all pairs of intersecting segments
        for t_segment, s_segment in timeline.co_iter(source.get_timeline()):
            # loop on all tracks
            for s_track in source.get_tracks(s_segment):
                # try to use source track name whenever possible
                t_track = tagged.new_track(
                    t_segment, candidate=s_track, prefix='')
                # get source label...
                source_label = source[s_segment, s_track]
                # ... and tag target segment with it
                tagged[t_segment, t_track] = source_label

        return tagged


class ConservativeDirectTagger(BaseTagger):
    """
    Conservative direct segment tagger

    Only supports annotation tagging.


    """

    def _tag_annotation(self, source, annotation):
        """Annotation tagging

        Parameters
        ----------
        source : Annotation
            Source annotation whose labels will be propagated
        annotation : Annotation
            Target annotation whose tracks will be tagged.

        Returns
        -------
        tagged : Annotation
            Tagged `annotation`.

        """

        # initialize tagged annotation as a copy of target annotation
        tagged = annotation.copy()

        # tag each segment of target annotation, one after the other
        for segment in tagged.itersegments():

            # extract the part of source annotation
            # intersecting current target segment
            t = source.crop(segment, mode='loose')

            # if there is no intersecting segment
            # just skip to the next one
            if not t:
                continue

            # only tag segment
            # when target has exactly one track and source only one
            # co-occurring label

            # don't do anything if target has more than one track
            tracks = tagged.get_tracks(segment)
            if len(tracks) > 1:
                continue
            else:
                track = tracks.pop()

            # don't do anything if source has more than one label
            labels = t.labels()
            if len(labels) > 1:
                continue
            else:
                label = labels[0]

            tagged[segment, track] = label

        return tagged


class ArgMaxDirectTagger(BaseTagger):
    """
    ArgMax direct segment tagger

    Parameters
    ----------
    unknown_last : bool
        If unknown_last is True,

    It supports both timeline and annotation tagging.

    **Timeline tagging.**
    Each segment in target timeline is tagged with the `N` intersecting labels
    with greatest co-occurrence duration:
        - `N` is set to 1 in case source annotation is single-track.
        - In case of a multi-track source, `N` is set to the the maximum number
          of simultaneous tracks in intersecting source segments.

    **Annotation tagging.**

    """

    def __init__(self, known_first=False):
        super(ArgMaxDirectTagger, self).__init__()
        self.known_first = known_first

    def _tag_timeline(self, source, timeline):
        """Timeline tagging

        Each segment in target `timeline` is tagged with the `N` intersecting
        labels with greatest co-occurrence duration.
        `N` is set to the the maximum number of simultaneous tracks in
        intersecting source segments.

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

        # initialize tagged timeline as an empty copy of source
        T = source.empty()

        # track name
        n = 0

        # tag each segment of target timeline, one after the other
        for segment in timeline:

            # extract the part of source annotation
            # intersecting current target segment
            t = source.crop(segment, mode='loose')

            # if there is no intersecting segment
            # just skip to the next one
            if not t:
                continue

            # find largest number of co-occurring tracks ==> N
            # find N labels with greatest intersection duration
            # tag N tracks with those N labels

            # find largest number of simultaneous tracks (n_tracks)
            n_tracks = max([len(t.get_tracks(s)) for s in t.itersegments()])

            # find n_tracks labels with greatest intersection duration
            # and add them to the segment
            for i in range(n_tracks):

                # find current best label
                label = t.argmax(segment, known_first=self.known_first)

                # if there is no label in stock
                # just stop tagging this segment
                if not label:
                    break
                # if current best label exists
                # create a new track and go for it.
                else:
                    T[segment, n] = label
                    n = n+1
                    t = t.subset(set([label]), invert=True)

        return T

    def _tag_annotation(self, source, annotation):
        """Annotation tagging

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

        # initialize tagged annotation as a copy of target annotation
        tagged = annotation.copy()

        # tag each segment of target annotation, one after the other
        for segment in tagged.itersegments():

            # extract the part of source annotation
            # intersecting current target segment
            t = source.crop(segment, mode='loose')

            # if there is no intersecting segment
            # just skip to the next one
            if not t:
                continue

            # tag each track one after the other
            # always choose label with greatest intersection duration
            for track in tagged.get_tracks(segment):

                # find current best label
                label = t.argmax(segment, known_first=self.known_first)

                # if there is no label in stock
                # just stop tagging this segment
                if not label:
                    break

                # if current best label exists,
                # go for it and tag track
                else:
                    tagged[segment, track] = label
                    t = t.subset(set([label]), invert=True)

        return tagged


if __name__ == "__main__":
    import doctest
    doctest.testmod()
