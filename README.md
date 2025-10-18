# Connectionist Temporal Classification

This repo implements the [CTC paper](https://www.cs.toronto.edu/~graves/icml_2006.pdf) from 2006 in JAX.

### Download the TIMIT dataset.

https://github.com/philipperemy/timit which points to the academictorrents [link](
https://academictorrents.com/details/34e2b78745138186976cbc27939b1b34d18bd5b3)

### Prepare the TIMIT dataset for training.

`sh prepare_dataset.sh`

You need to change BASE_DIR and OUTPUT_DIR in above script.

### Train Bi-LSTM network with CTC.

`sh run_training.sh`

Supply the appropriate args in the above script. It trains to 46% label error rate
on the reduced phone set. Reduced phone set uses 42 labels (41 phones + 1 blank label).
I couldn't quite get it down to the 24% label error rate as claimed in Alex Graves'
2nd paper ([Phoneme recognition in TIMIT with BiLSTM-CTC](https://arxiv.org/abs/0804.3269)).

My results should be reproducible however. I spent a looong time on this (probably 100+ hrs)
and don't have the patience to improve it at the moment. I'm not quite sure why I
can't reproduce the results. If you are looking at this code and find something,
please feel free to raise a pull request !

### Run the unit-tests.

`bazel test :all`

The CTC loss and its gradient computation is implemented in ctc.py. This is
purely for understanding purposes. The correctness of the implementation is
verified through the tests, which asserts some invariants and compares the loss
value and grads with optax's implementation.