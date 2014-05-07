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

from pyannote.core import T, TStart
import re
import networkx as nx


class WordsToSentencesAlignment(object):
    """
    Parameters
    ----------
    punctuation : boolean, optional
    """

    def __init__(self, punctuation=True):
        super(WordsToSentencesAlignment, self).__init__()
        self.punctuation = punctuation

    def _clean_sentence(self, sentenceWords):
        sentenceClean = re.sub(r'\([^\)]+\)','', sentenceWords)
        sentenceClean = re.sub(r'\[[^\]]+\]','', sentenceClean)
        #sentenceClean = re.sub('[.!,;?":]','', sentenceClean)
            
        sentenceClean = re.sub(r'^[\.!,;?":]+','', sentenceClean)

        sentenceClean = re.sub(r'([\.!,;?":]+)[ ]+([\.!,;?":]+)','\g<1>\g<2>', sentenceClean)
        sentenceClean = re.sub(r'[ ]*([\.!,;?":]+)','\g<1> ', sentenceClean)

        sentenceClean = re.sub(r' +',' ', sentenceClean)
        sentenceClean = sentenceClean.strip()
        return sentenceClean
    
    def __call__(self, words, sentences):
        """

        Parameters
        ----------
        words : `pyannote.core.Transcription`
        sentences : `pyannote.core.Transcription`

        Returns
        -------
        sentences : `pyannote.core.Transcription`


        """
        lastIndexNode=0

        end = False
        
        T.reset()
        sentences, mapping = sentences.relabel_drifting_nodes()
        words, _ = words.relabel_drifting_nodes()
        sentences, mapping = sentences.relabel_drifting_nodes(mapping=mapping)

        nodesWords = nx.topological_sort(words)
        if nodesWords[lastIndexNode] == TStart:
            lastIndexNode += 1

        last = -1
        next = -1

        first_node = None

        first = -1

        for t1, t2, data in sentences.ordered_edges_iter(data=True):
            
            if 'speech' not in data:
                continue

            sentence = data['speech']
            speaker = data['speaker']
            sentenceClean = self._clean_sentence(sentence)
            
            if not self.punctuation:
                sentenceClean = re.sub(r'[\.!,;?":]+','', sentenceClean)

            if sentenceClean != "":

                sentenceWords = ""
            
                if lastIndexNode < len(nodesWords):

                    if first_node is None and t1 != TStart:
                        first_node = t1
                        sentences.add_edge(first_node, nodesWords[lastIndexNode])

                    node_manual_trs_start = t1
                    node_manual_trs_end = t2

                    node_float = T()
                    remainingData = None
                    if last > 0 and next > 0:
                        for key in words[last][next]:
                            dataWord = words[last][next][key]
                            if 'speech' in dataWord:
                                remainingData = dataWord
                                sentenceWords = remainingData['speech']
                                sentenceWords = self._clean_sentence(sentenceWords)
                                last = -1
                                next = -1
                    
                    bAlreadyAdded = False

                    if(remainingData is not None):
                        if 'speech' in remainingData:
                            remainingData['speaker']=speaker
                        sentences.add_edge(node_manual_trs_start, nodesWords[lastIndexNode], **remainingData)
                        if sentenceWords == sentenceClean:
                            sentences.add_edge(nodesWords[lastIndexNode], node_manual_trs_end)
                            bAlreadyAdded = True

                    if not bAlreadyAdded:
                        if not sentences.has_edge(node_manual_trs_start, nodesWords[lastIndexNode]):
                            sentences.add_edge(node_manual_trs_start, nodesWords[lastIndexNode])

                        node_end = ""
                        previousNode = None
                        while not end and lastIndexNode < len(nodesWords):
                            node = nodesWords[lastIndexNode]
                            for node2 in sorted(words.successors(node)):
                                
                                node_start = node
                                node_end = node2
                                
                                if previousNode is not None:
                                    if not sentences.has_edge(previousNode, node_start) and previousNode != node_start :
                                        sentences.add_edge(previousNode, node_start)

                                for key in words[node][node2]:
                                    dataWord = words[node][node2][key]
                                    if 'speech' in dataWord:
                                        dataWord['speaker']=speaker
                                    sentences.add_edge(node_start, node_end, **dataWord)
                                
                                    if 'speech' in dataWord:
                                        if sentenceWords == "":
                                            sentenceWords = dataWord['speech']
                                        else:
                                            sentenceWords += " " + dataWord['speech']
                                        sentenceWords = self._clean_sentence(sentenceWords)
                                if sentenceWords == sentenceClean:
                                    if re.search(r'[\.!,;?":]$', sentenceClean):
                                        #Have to add the next anchored just before the end of the speech turn ...
                                        lastIndexNode+= 2
                                        if lastIndexNode < len(nodesWords):
                                            node = nodesWords[lastIndexNode]
                                            if node.anchored:
                                                sentences.add_edge(node_end, node)
                                                node_end = node
                                                lastIndexNode -= 1
                                            else:
                                                lastIndexNode -= 2
                                    end = True
                                previousNode = node_end
                            lastIndexNode+=1

                        if lastIndexNode+1 < len(nodesWords):
                            last = nodesWords[lastIndexNode]
                            next = nodesWords[lastIndexNode+1]

                        #print "%s -> %s" % (node_end, node_manual_trs_end)
                        lastIndexNode+=1

                        sentences.add_edge(node_end, node_manual_trs_end)
                        
                        end = False

                elif sentenceClean != "":
                    print "Unable to align '%s' !" % (sentenceClean)
                    return None

        return sentences