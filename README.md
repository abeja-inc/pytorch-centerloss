# pytorch-centerloss

This repository contains simple and experimental implementation of `Temporal Ensembling` [1] and some of its variants in PyTorch.

Here we briefly explain our novel method `Improved Temporal Ensembling` or TE++, which modify `Temporal Ensembling` by changing regularization term of `Temporal Ensembling` (we also refert to it center loss).

While in `Temporal Ensembling`, regularization is defined as the gap between the current estimation `p(y|x)` and the histrical moving average of `p(y|x)`, our `Improved Temporal Ensembling` take the gap as that between the `log p(y|x)` and the historial moving average of `log p(y|x)`.

The intuition behind this modification is the WAIC (Widely Applicable Information Criteria, which is the generalization of well-known AIC) theory [2].

According to the WAIC theory, any bayesian model p(x, y) has empirical measurement called WAIC, the
expectation value of which is asymptotically equal to the average Bayes generalization loss.
WAIC can be computed without knowing real data distribution and defined as

```
WAIC = -(empirical log likelihood) + (functional variance)
```

where N is dataset size.
Functional variance represents the fluctuation of `log p(x, y)` against the average of it and is defined as
```
functional variance = E[(log p(x, y)  - (avg of log p(x, y)))^2].
```
where estimation/average E is taken over empirical data distribution of (x, y).

Considering this WAIC theory, we propose to replace the regularization term of `temporal ensembling` with the `empirical functional variance`. Our TE++ is a practical implementation of this formulation.

# Example

We prepared training code for cifar10.

```
python train-ssc.py <expname>
```
where <expname> is any string which will appear in the `runs' area of in Tensorboard.


# references

- [1] [S. Laine+, "Temporal Ensembling for Semi-Supervised Learning", 2016](https://arxiv.org/abs/1610.02242)
- [2] [S. Watanabe, "Asymptotic Equivalence of Bayes Cross Validation and Widely Applicable Information Criterion in Singular Learning Theory", 2010](http://www.jmlr.org/papers/volume11/watanabe10a/watanabe10a.pdf)