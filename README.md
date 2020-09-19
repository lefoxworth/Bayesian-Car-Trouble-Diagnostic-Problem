This Bayesian Network program prompts the user for a query hypothesis and calculates the inferred probability.

Data for prior/conditional probabilities are read from a data file ('data' in the 'homework1' directory).
This file is a list of floats, categorized by line, and split by a single whitespace.
The corresponding variables of each line are as follows:
IW: [prior prob]
B: [IW prob T] [IW prob F]
SM: [IW prob T] [IW prob F]
R: [B prob T] [B prob F]
I: [B prob T] [B prob F]
G: [prior prob]
S: [I prob T] [I prob F] [SM prob T] [SM prob F] [G prob T] [ G prob F]
M: [S prob T] [S prob F]

Ex., ".9 .3" on the last line of 'data' corresponds to M with a probability of .9 given S=True, and .3 given S=False.

This code is based on the framework given in the aima-python Jupyter notebook, which can be found at:
https://github.com/aimacode/aima-python/blob/master/probability4e.ipynb
