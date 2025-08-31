# Connectionist Temporal Classification

This repo implements the [CTC paper](https://www.cs.toronto.edu/~graves/icml_2006.pdf) from 2006 in JAX.

### Run the unit-tests

`bazel test :all`

The CTC loss and its gradient computation is implemented in ctc.py. This is
purely for understanding purposes. The correctness of the implementation is
verified through the tests, which asserts some invariants and compares the loss
value and grads with optax's implementation.