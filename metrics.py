"""Compute essential eval metrics."""

import numpy as np


def collapse_repetitions(predicted_label_seq, blank_id):
    collapsed_seq = [predicted_label_seq[0]]
    for lbl in predicted_label_seq[1:]:
        if lbl != collapsed_seq[-1]:
            collapsed_seq.append(lbl)
    collapsed_seq = [lbl for lbl in collapsed_seq if lbl != blank_id]
    return collapsed_seq


def label_error(predicted_label_seq, label_seq, blank_id):
    """Computes label error which is the edit distance after collapsing repetitions and removing blanks."""

    def edit_distance(y1, y2):
        m = len(y1)
        n = len(y2)
        if m == 0 and n > 0:
            return n
        if n == 0 and m > 0:
            return m
        if n == 0 and m == 0:
            return 0

        e = np.zeros((m, n), dtype="int")
        e[0, 1:] = np.arange(1, n)
        e[1:, 0] = np.arange(1, m)
        for i in range(1, m):
            for j in range(1, n):
                local_cost = 0 if y1[i] == y2[j] else 1
                e[i, j] = min(e[i - 1, j - 1], e[i - 1, j], e[i, j - 1]) + local_cost
        return e[-1, -1]

    collapsed_predicted_seq = collapse_repetitions(predicted_label_seq, blank_id)
    return edit_distance(collapsed_predicted_seq, label_seq)
