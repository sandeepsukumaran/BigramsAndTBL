# BigramsAndTBL
Computation of various bigrams models, Naive Bayesian Part of Speech tagging, and Transformation Based Learner

## Bigram Models
BigramProbabilities.py reads from a corpus and calculates the bigram model (counts and probabilities) for three cases:
1. No smoothing
2. Laplacian smoothing
3. Good-Turing discounting based smoothing (no regression)

The bigram models are written to separate files.

## Naive Bayesian POS tagging:
NaiveBayesian.py reads from a corpus and computes model parameters of a Hidden Markov Model.
It also prints computations for use elsewhere.

## Transformation Based Learning:
TBL.py reads from a corpus and runs Brill's tagging on a very narrow set of template possibilities.
Only templates of type "Change from_tag to to_tag when previous is prev_tag." are considered and only rules involving NN and VB tags are computed.

## Software
Built and tested on Python3.6
No additional dependencies.
