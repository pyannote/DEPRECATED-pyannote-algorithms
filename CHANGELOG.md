### Version 0.8 (2018-05-22)

  - setup: switch to sortedcollections 1.x
  - setup: switch to pyannote.core 1.3.x
  - fix: fix support for networkx 2.x

### Version 0.7.3 (2017-02-19)

  - fix: fix clustering constraints
  - chore: simplify HACModel API

### Version 0.7.2 (2017-02-05)

  - improve: faster hierarchical agglomerative clustering
  - setup: switch to pyannote.core 0.13

### Version 0.6.7 (2016-11-29)

  - fix: fix greedy label mapper

### Version 0.6.6 (2016-11-08)

 - fix: fix hierarchical agglomerative clustering
 - setup: update to pyannote.core 0.8

### Version 0.6.5 (2016-06-24)

  - setup: update to pyannote.core 0.6.6

### Version 0.6.4 (2016-04-08)

  - feat: add max_gap parameters to linear BIC clustering

### Version 0.6.3 (2016-04-04)

  - feat: greedy mapping (fast approximate hungarian mapping)

### Version 0.6.2 (2016-03-30)

  - improve: faster hungarian mapping (divide-and-conquer)

### Version 0.6.1 (2016-03-29)

  - python 3 support
  - fix: remove broken wip code

### Version 0.5.5 (2016-03-28)

  - fix: corner case in hierarchiccal agglomerative clustering

### Version 0.5.4 (2016-03-21)

  - improve: faster constrained hierachichal agglomerative clustering

### Version 0.5.3 (2016-02-23)

  - fix: bug in {ConservativeDirect|ArgMax}Mapper

### Version 0.5.2 (2016-02-19)

  - fix: forgot to refactor {Complete|Average|Single}LinkageClustering
  - refactor: mappers now rely on xarray cooccurrence matrix

### Version 0.5 (2016-02-12)

  - refactor: complete refactoring of hierarchical agglomerative clustering
  - chore: deprecates 'thresh' parameter in favor of 'tol'

### Version 0.4.6 (2015-03-20)

  - feat: new methods add to GMMClassification (score and LLR)

### Version 0.4.4 (2015-03-16)

  - fix: inherit GMMUBMClassification.predict_proba
  - setup:

### Version 0.4.1 (2015-02-26)

  - fix: corner case in Viterbi consecutive constraint

### Version 0.4.0 (2015-02-25)

 - feat(segmentation): add support for input segmentation
 - improve(LBG): GMM iterator + better default parameters
 - feat: LLRIsotonicRegression IPython display

### Version 0.3.2 (2015-01-26)

  - fix: GMMClassification supports Timeline input

### Version 0.3.1 (2015-01-23)

  - feat: BIC segmentation algorithm
  - refactor: duration-constrained Viterbi

### Version 0.2.1 (2015-01-14)

  - fix: missing 'equal_priors' option
  - fix: GMM-based classification without score calibration

### Version 0.2 (2015-01-07)

  - feat(classification): GMM- and GMM/UBM-based classification
  - feat(segmentation): constrained Viterbi segmentation
  - improve: improve log-likelihood ratio calibration
  - setup: switch to pyannote.core 0.3+

### Version 0.1 (2014-10-30)

  - feat: HMM classification (with Viterbi decoding)

### Version 0.0.4.1 (2014-10-01)

  - fix: remove no longer existing import

### Version 0.0.4 (2014-08-05)

  - feat(alignment): add transcription alignment

### Version 0.0.3 (2014-07-08)

  - fix(stats): fix estimation of scores range

### Version 0.0.2 (2014-07-08)

  - feat(segmentation): add algorithms based on sliding windows
  - feat(alignment): add generic dynamic time warping implementation
  - feat(stats): add log-likelihood ratio linear/isotonic regression
  - feat(stats): add Linde–Buzo–Gray algorithm for GMM estimation

### Version 0.0.1 (2014-05-11)

  - feat: hierarchical agglomerative clustering
  - feat: BIC clustering
