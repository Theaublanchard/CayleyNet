# CayleyNets
We present a Pytorch-geometric implementation of the Graph Convolutional Neural Network illustrated in:

CayleyNets: Graph Convolutional Neural Networks with Complex Rational Spectral Filters<br>
IEEE Transactions on Signal Processing, 2018<br>
Ron Levie*, Federico Monti*, Xavier Bresson, Michael M. Bronstein

https://arxiv.org/abs/1705.07664

The repository also contain a sparse implementation of the Jacobi method.

The performance of this convolutional layer are illustred on the CORA dataset.
Rational spectral filters are approximated with Jacobi Method to provide an efficient solution.

## When shall I use CayleyNet?

CayleyNet is a Graph CNN with spectral zoom properties able to effectively operate with signals defined over graphs. Thanks to its particular spectral properties, CayleyNet is able to work with both long and short range frequency bands. It is versatile enough to be used in a variety of context from vertex classification to community detection and matrix completion. It achieved close state-of-the-art performances on such tasks according to the authors of the original paper.