"""Tests for verifying correctness of forward-backward algorithm."""

import unittest

import ctc
import numpy as np
from scipy.special import logsumexp, log_softmax
import optax
import jax
import jax.numpy as jnp


class TestForwardBackward(unittest.TestCase):
    def test_probability_invariant(self):
        np.random.seed(123)
        x_tv = np.random.normal(0.0, 10000, (9, 4))
        logprobs_tv = log_softmax(x_tv, axis=-1)
        y = np.array([0, 1, 1, 2, 2])
        (
            log_posterior_probs,
            _,
        ) = ctc.compute_posterior_probs(logprobs_tv, y, 3, -100000.0)
        self.assertTrue(np.all(log_posterior_probs <= 0))
        self.assertTrue(
            np.allclose(logsumexp(log_posterior_probs, axis=-1, keepdims=True), 0.0)
        )

    def test_forward_backward_posterior_probs_same_at_all_time_steps(self):
        np.random.seed(123)
        x_tv = np.random.normal(0.0, 10000, (9, 4))
        logprobs_tv = log_softmax(x_tv, axis=-1)
        y = np.array([0, 1, 1, 2, 2])
        log_alpha = ctc.forward_algo(logprobs_tv, y, 3, -100000.0)
        log_beta = ctc.backward_algo(logprobs_tv, y, 3, -100000.0)

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
        x_tv_logits = np.random.normal(loc=0.0, scale=1, size=(1, 9, 4))
        blank_id = 3

        # optax actually just computes the NLL of the target sequence.
        loss, grad = ctc.compute_ctc_optax_equiv(
            x_tv_logits[0], y, blank_id, log_epsilon=-100000.0
        )

        def loss_fn(logits, y, blank_id, log_epsilon):
            logit_paddings = np.zeros((1, logits.shape[1]))
            label_paddings = np.zeros((1, y.shape[0]))
            return jnp.mean(
                optax.ctc_loss(
                    logits=logits,
                    logit_paddings=logit_paddings,
                    labels=y[np.newaxis, ...],
                    label_paddings=label_paddings,
                    blank_id=blank_id,
                    log_epsilon=log_epsilon,
                )
            )

        expected_loss, expected_grads = jax.value_and_grad(loss_fn)(
            x_tv_logits, y, blank_id, -100000.0
        )

        self.assertAlmostEqual(loss, expected_loss, places=5)
        self.assertTrue(np.allclose(grad, expected_grads[0], atol=1e-6))


if __name__ == "__main__":
    unittest.main()
