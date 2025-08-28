"""Implements the CTC objective as described in the CTC paper."""

import numpy as np
from scipy.special import logsumexp

# X is a sequence of inputs [x1, x2, ...., xM]. Y is a sequence of symbols
# [y1, y2, ..., yN], where N <= M. The CTC objective is used when the alignment
# between X and Y is not provided.
#
# The CTC objective for a single (X, Y) pair maximizes the log-likelihood
# log P(Y|X) over all valid alignments.
# where P(Y|X) = \sum_{A \in \textbf{A}_{X,Y}} \product_{t=1}^T p_t(a_t | X)
# where A is a valid alignment [a1, a2, ...aM] between X and Y, and
# \textbf{A}_{X,Y} is the set of all valid alignments between X and Y.
#
# A few resources I followed to understand CTC and the forward-backward
# algorithm.
# - 1) The distillpub article on CTC is excellent: https://distill.pub/2017/ctc/
# - 2) Prof. Bhiksha Raj's lecture: https://youtu.be/xwdBvlpZvXU

# Forward Algorithm
# Initialization:
# - \alpha(0, 0) = x[0, y[0]]
# - \alpha(0, r) = 0 for r > 0
# for t = 1 to T - 1
#  - \alpha(t, 0) = \alpha(t-1, 0) * x[t, y[0]]
#  - for l = 1 to K - 1
#     - \alpha(t, l) = (\alpha(t-1, l) + \alpha(t-1, l-1)) * x[t, y[l]]

# skips are permitted across a blank, but only if the symbols on either side are
# different.


def create_extended_sequence(y, blank_idx):
    K = y.shape[0]
    s = np.full((2 * K + 1,), blank_idx, dtype=np.int32)
    s[1::2] = y
    return s


def forward_algo(
    x_tv: np.array,
    y: np.array,
    blank_idx: int,
    log_epsilon: float,
):
    """Forward computation of the forward-backward algo."""
    T = x_tv.shape[0]
    K = y.shape[0]

    N = 2 * K + 1
    s = create_extended_sequence(y, blank_idx)

    # initialize alpha of shape [T, N].
    log_alpha = [[log_epsilon] * N for _ in range(T)]
    log_alpha[0][0] = np.log(x_tv[0, blank_idx])
    log_alpha[0][1] = np.log(x_tv[0, s[1]])

    for t in range(1, T):
        log_alpha[t][0] = log_alpha[t - 1][0] + np.log(x_tv[t, s[0]])  # s[0] is blank
        for k in range(1, N):
            log_alpha[t][k] = logsumexp([log_alpha[t - 1][k], log_alpha[t - 1][k - 1]])
            if k >= 2 and s[k] != s[k - 2]:
                log_alpha[t][k] = logsumexp([log_alpha[t][k], log_alpha[t - 1][k - 2]])
            log_alpha[t][k] = log_alpha[t][k] + np.log(x_tv[t, s[k]])

    return np.array(log_alpha)


def backward_algo(
    x_tv: np.array,
    y: np.array,
    blank_idx: int,
    log_epsilon: float,
    # discount_current: bool = True,
):
    """Backward computation of the forward-backward algo."""
    T = x_tv.shape[0]
    K = y.shape[0]
    N = 2 * K + 1
    s = create_extended_sequence(y, blank_idx)

    # initialize beta of shape [T, N]
    log_beta = [[log_epsilon] * N for _ in range(T)]
    log_beta[T - 1][N - 1] = 0.0
    log_beta[T - 1][N - 2] = 0.0

    for t in range(T - 2, -1, -1):
        for k in range(N - 1, -1, -1):
            candidates = []
            # stay in same state
            candidates.append(log_beta[t + 1][k] + np.log(x_tv[t + 1, s[k]]))

            # transition to next state
            if k + 1 < N:
                candidates.append(
                    log_beta[t + 1][k + 1] + np.log(x_tv[t + 1, s[k + 1]])
                )
            if k + 2 < N and s[k] != s[k + 2]:
                candidates.append(
                    log_beta[t + 1][k + 2] + np.log(x_tv[t + 1, s[k + 2]])
                )
            log_beta[t][k] = logsumexp(candidates)

    return np.array(log_beta)


def compute_posterior_probs(
    x_tv: np.array, y: np.array, blank_idx: int, log_epsilon: float = -100000.0
):
    log_alpha_tr = forward_algo(x_tv, y, blank_idx, log_epsilon)
    log_beta_tr = backward_algo(x_tv, y, blank_idx, log_epsilon)

    # posterior probs
    log_gamma_tr = log_alpha_tr + log_beta_tr
    norms = logsumexp(log_gamma_tr, axis=-1, keepdims=True)
    log_gamma_tr = log_gamma_tr - norms

    return log_gamma_tr


def compute_ctc(x_tv: np.array, y: np.array, blank_idx: int):
    log_gamma_tr = compute_posterior_probs(x_tv, y, blank_idx)
    s = create_extended_sequence(y, blank_idx)
    divergence = -np.exp(log_gamma_tr) * np.log(x_tv[:, s])
    return divergence.sum()


# def compute_ctc_grad(x_tv: np.array, y_t: np.array, blank_idx: int):
