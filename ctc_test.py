"""Tests for verifying correctness of forward-backward algorithm."""

import unittest

import ctc
import numpy as np
from scipy.special import logsumexp
import optax


class TestForwardBackward(unittest.TestCase):
    def test_probability_invariant(self):
        np.random.seed(123)
        x_tv = np.random.dirichlet(alpha=[1.0] * 4, size=5)
        y = np.array([0, 1, 1, 2, 2])
        log_posterior_probs = ctc.compute_posterior_probs(x_tv, y, 3, -100000.0)
        self.assertTrue(np.all(log_posterior_probs <= 0))
        self.assertTrue(
            np.allclose(logsumexp(log_posterior_probs, axis=-1, keepdims=True), 0.0)
        )

    def test_forward_backward_posterior_probs_same_at_all_time_steps(self):
        np.random.seed(123)
        x_tv = np.random.dirichlet(alpha=[1.0, 1.0, 1.0, 1.0], size=5)
        y = np.array([0, 1, 1, 2, 2])
        log_alpha = ctc.forward_algo(x_tv, y, 3, -100000.0)
        log_beta = ctc.backward_algo(x_tv, y, 3, -100000.0)

        log_gamma = log_alpha + log_beta

        log_prob_disjoint_paths = logsumexp(log_gamma, axis=-1, keepdims=True)
        self.assertTrue(
            np.allclose(log_prob_disjoint_paths, log_prob_disjoint_paths[0])
        )


class TestCTC(unittest.TestCase):
    def stable_softmax(self, x, axis):
        x_max = np.max(x, axis=axis, keepdims=True)
        x = x - x_max
        x_exp = np.exp(x)
        return x_exp / x_exp.sum(axis=axis, keepdims=True)

    def test_compare_ctc_loss_with_optax_impl(self):
        np.random.seed(123)
        y = np.array([0, 1, 1, 2, 2])
        # for target sequence with repetitions and of length N,we need minimum
        # 2*N-1 input sequence length. N for the actual symbols itself and N-1
        # for the blanks between repeated symbols.
        x_tv_logits = np.random.normal(loc=0.0, scale=0.01, size=(1, 9, 4))
        x_tv_probs = self.stable_softmax(x_tv_logits, axis=-1)

        blank_id = 3

        # optax actually just computes the NLL of the target sequence. For this,
        # we need to compare only the log_probs from alpha.
        log_alpha = ctc.forward_algo(x_tv_probs[0], y, blank_id, log_epsilon=-100000.0)

        #
        logit_paddings = np.zeros((1, x_tv_logits.shape[1]))
        label_paddings = np.zeros((1, 5))
        expected_loss = optax.ctc_loss(
            logits=x_tv_logits,
            logit_paddings=logit_paddings,
            labels=y[np.newaxis, ...],
            label_paddings=label_paddings,
            blank_id=blank_id,
            log_epsilon=-100000.0,
        )

        self.assertAlmostEqual(
            -logsumexp(log_alpha[-1, -2:]), expected_loss[0, ...], places=5
        )


if __name__ == "__main__":
    unittest.main()
