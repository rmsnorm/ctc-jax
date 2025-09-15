import jax.numpy as jnp
from flax import nnx


class LSTMCell(nnx.Module):
    def __init__(self, input_dim: int, hidden_dim: int, rngs: nnx.Rngs):
        H = hidden_dim + input_dim
        self.gate = nnx.Einsum(
            "bH, Hh -> bh",
            kernel_shape=(H, 4 * hidden_dim),
            bias_shape=4 * hidden_dim,
            param_dtype=jnp.float32,
            rngs=rngs,
        )

    def __call__(self, x_bd, h_bh, c_bh):
        x_bH = jnp.column_stack([x_bd, h_bh])
        gate_out = self.gate(x_bH)

        f, i, o, g = jnp.split(gate_out, 4, axis=-1)
        forget_gate = nnx.sigmoid(f)
        input_gate = nnx.sigmoid(i)
        output_gate = nnx.sigmoid(o)
        cell_gate = nnx.tanh(g)

        cell_bh = forget_gate * c_bh + input_gate * cell_gate
        state_bh = output_gate * nnx.tanh(cell_bh)
        return state_bh, cell_bh


class LSTM(nnx.Module):
    def __init__(self, input_dim, hidden_dim: int, rngs: nnx.Rngs):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.cell = LSTMCell(input_dim, hidden_dim, rngs)
        self.h_init = nnx.Param(
            nnx.initializers.glorot_normal()(rngs.params(), (1, hidden_dim))
        )

    def __call__(self, x_btd, input_paddings):
        del input_paddings
        b = x_btd.shape[0]

        @nnx.scan
        def lstm_scan(carry, x):
            h_bh, c_bh = carry
            x_bd = x
            h_bh, c_bh = self.cell(x_bd, h_bh, c_bh)
            return (h_bh, c_bh), h_bh

        c_bh = jnp.zeros((b, self.hidden_dim))
        h_bh = jnp.tile(self.h_init, (b, 1))
        carry = (h_bh, c_bh)
        _, h_tbh = lstm_scan(carry, x_btd.transpose(1, 0, 2))
        return h_tbh.transpose(1, 0, 2)


class BiLSTM(nnx.Module):
    def __init__(self, input_dim: int, hidden_dim: int, rngs: nnx.Rngs):
        self.fwd_lstm = LSTM(input_dim, hidden_dim, rngs)
        self.rev_lstm = LSTM(input_dim, hidden_dim, rngs)

    def __call__(self, x_btd, input_paddings):
        fwd_h_bth = self.fwd_lstm(x_btd, input_paddings)
        lengths_b = jnp.sum(input_paddings == 0.0, axis=-1).astype(jnp.int32)[:, None]
        flipped_x_btd = self._flip_input(x_btd, lengths_b)
        rev_h_bth = self.rev_lstm(flipped_x_btd, input_paddings)
        rev_h_bth = self._flip_input(rev_h_bth, lengths_b)

        h_bth = jnp.concatenate((fwd_h_bth, rev_h_bth), axis=-1)

        return h_bth

    def _flip_input(self, x_btd, lengths_b):
        b, t, d = x_btd.shape
        indices = jnp.tile(jnp.arange(t)[None, :], (b, 1))
        reversed_indices = lengths_b - 1 - indices
        valid_mask = indices < lengths_b
        final_indices = jnp.where(valid_mask, reversed_indices, indices)
        batch_idx = jnp.arange(b)[:, None]
        return x_btd[batch_idx, final_indices]


class Network(nnx.Module):
    def __init__(
        self,
        is_bilstm: bool,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        rngs: nnx.Rngs,
    ):
        if is_bilstm:
            self.rnn = BiLSTM(input_dim, hidden_dim, rngs)
            self.output_head = nnx.Linear(2 * hidden_dim, output_dim, rngs=rngs)
        else:
            self.rnn = LSTM(input_dim, hidden_dim, rngs)
            self.output_head = nnx.Linear(hidden_dim, output_dim, rngs=rngs)

    @nnx.jit
    def __call__(self, x_btd, input_paddings):
        h_bth = self.rnn(x_btd, input_paddings)
        logits_btv = self.output_head(h_bth)
        return logits_btv
