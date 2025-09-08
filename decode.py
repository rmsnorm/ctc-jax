"Implements decoding schemes: best_path_decoding and prefix search decoding."

import numpy as np


def best_path_decode(logprobs_btv, input_paddings_bt):
    b, T, v = logprobs_btv.shape

    best_path_bt = np.argmax(logprobs_btv, axis=-1)
    valid_mask = (input_paddings_bt == 0.0).astype(np.float32)

    best_path_logprobs = np.sum(
        (logprobs_btv * valid_mask[..., None])[best_path_bt], axis=-1
    )

    end_idx_b = np.argmax(input_paddings_bt == 1.0, axis=-1) - 1
    best_paths = []

    for i in range(b):
        T_i = end_idx_b[i]
        best_path = best_path_bt[i, :T_i].tolist()
        best_paths.append((best_path, best_path_logprobs[i]))
    return best_paths
