# Connectionist Temporal Classification

This repo implements the [CTC paper](https://www.cs.toronto.edu/~graves/icml_2006.pdf) from 2006 in JAX.

### Download the TIMIT dataset.

https://github.com/philipperemy/timit which points to the academictorrents [link](
https://academictorrents.com/details/34e2b78745138186976cbc27939b1b34d18bd5b3)

### Prepare the TIMIT dataset for training.

`sh prepare_dataset.sh`

You need to change BASE_DIR and OUTPUT_DIR in above script.

### Train Bi-LSTM network with CTC.
 
### Eval the trained network.

### Run the unit-tests.

`bazel test :all`

The CTC loss and its gradient computation is implemented in ctc.py. This is
purely for understanding purposes. The correctness of the implementation is
verified through the tests, which asserts some invariants and compares the loss
value and grads with optax's implementation.