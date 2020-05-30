# Bayesian NMF with TensorFlow Probability

## How to run
1. Get `ml-100k` dataset from https://grouplens.org/datasets/movielens/100k/ and place it under `data`.
2. run 

```
$ docker build -t nmf_tfp:0.1 .
$ docker run -ti nmf_tfp:0.1 /bin/bash
# python tfp_nmf.py
```