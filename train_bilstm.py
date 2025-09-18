"""Train BiLSTM with CTC loss on prepared dataset."""

import argparse
import json
import data_loader
import lstm
import metrics
import decode
import tqdm
from flax import nnx
import optax
import jax.numpy as jnp
import tensorflow as tf
import wandb
import numpy as np
import jax
import phoneset

parser = argparse.ArgumentParser(
    prog="TrainBiLSTM",
    description="This binary trains a BiLSTM with CTC loss on a dataset",
)

parser.add_argument(
    "--train_tfr", help="Path of prepared train dataset in TFRecord format", type=str
)
parser.add_argument(
    "--tune_tfr", help="Path of prepared 5% tuning dataset in TFRecord format", type=str
)
parser.add_argument("--train_config", help="Path of training config.json", type=str)
parser.add_argument("--wandb_key", help="Wandb API key", type=str)

eval_metrics = nnx.MultiMetric(
    loss=nnx.metrics.Average("loss"),
    label_error_rate=nnx.metrics.Average("label_error_rate"),
)

train_metrics_history = {"train_loss": [], "train_ctc_loss": [], "train_l2_loss": []}
eval_metrics_history = {"test_loss": [], "test_label_error_rate": []}


PHN_2_LABEL = dict(zip(phoneset.PHONE_SET, range(1, len(phoneset.PHONE_SET) + 1)))
LABEL_2_PHN = dict([(v, k) for k, v in PHN_2_LABEL.items()])


def compute_loss(
    model: lstm.Network, input_btd, logit_paddings, label, label_paddings, l2_reg
):
    """Computes ctc loss."""
    logits_btv = model(input_btd, logit_paddings)
    ctc_loss = jnp.mean(
        optax.losses.ctc_loss(
            logits_btv,
            logit_paddings,
            label,
            label_paddings,
        )
    )

    # L2 regularization
    def mask_fn(path):
        # Apply weight decay to weights but not biases
        return (
            "kernel" in path[-1].key
            if hasattr(path[-1], "key")
            else "bias" not in str(path)
        )

    l2_loss = 0.0
    for path, param in jax.tree_util.tree_leaves_with_path(nnx.state(model, nnx.Param)):
        if mask_fn(path):
            l2_loss += jax.numpy.sum(param**2)

    total_loss = ctc_loss + l2_reg * l2_loss

    # Auxiliary information (not used for gradients)
    aux_info = {
        "ctc_loss": ctc_loss,
        "l2_loss": l2_loss,
        "total_loss": total_loss,
    }

    return total_loss, aux_info


def compute_metrics(
    model: lstm.Network, input_btd, logit_paddings, label, label_paddings
):
    logits_btv = model(input_btd, logit_paddings)
    loss = jnp.mean(
        optax.losses.ctc_loss(
            logits_btv,
            logit_paddings,
            label,
            label_paddings,
        )
    )

    bsz = input_btd.shape[0]
    label_errors = 0

    logprobs_btv = nnx.log_softmax(logits_btv)
    best_paths_probs = decode.best_path_decode(logprobs_btv, logit_paddings)

    label_seq_total = 0
    for i in range(bsz):
        best_path, _ = best_paths_probs[i]
        label_seq = label[i]
        label_seq = label_seq[label_paddings[i] == 0.0]
        label_seq_total += label_seq.shape[0]
        label_errors += metrics.label_error(best_path, label_seq, blank_id=0)
    label_errors /= label_seq_total

    return loss, label_errors


@nnx.jit
def train_step(
    model, optimizer, input_btd, logit_paddings, label, label_paddings, l2_reg
):
    grad_fn = nnx.value_and_grad(compute_loss, has_aux=True)
    (loss, aux_info), grads = grad_fn(
        model, input_btd, logit_paddings, label, label_paddings, l2_reg
    )
    optimizer.update(grads)
    return loss, aux_info


def eval_step(model, input_btd, logit_paddings, label, label_paddings, nnx_metrics):
    loss, label_error_rate = compute_metrics(
        model, input_btd, logit_paddings, label, label_paddings
    )
    nnx_metrics.update(loss=loss, label_error_rate=label_error_rate)


def train_one_epoch(
    model,
    optimizer,
    train_data_loader,
    total_train_records,
    epoch,
    num_epochs,
    batch_size,
    l2_reg,
):
    total_train_steps = total_train_records // batch_size
    bar_format = "{desc}[{n_fmt}/{total_fmt}]{postfix} [{elapsed}<{remaining}]"
    model.train()  # Set model to the training mode: e.g. update batch statistics
    with tqdm.tqdm(
        desc=f"[train] epoch: {epoch}/{num_epochs}, ",
        total=total_train_steps,
        bar_format=bar_format,
        leave=True,
    ) as pbar:
        for i in range(total_train_steps):
            batch = train_data_loader.get_batch()
            loss, aux_info = train_step(
                model,
                optimizer,
                jnp.array(batch["input_seq"]),
                jnp.array(batch["input_paddings"]),
                jnp.array(batch["label"]),
                jnp.array(batch["label_paddings"]),
                l2_reg,
            )
            train_ctc_loss = aux_info["ctc_loss"]
            train_l2_loss = aux_info["l2_loss"]
            train_metrics_history["train_loss"].append(loss.item())
            train_metrics_history["train_ctc_loss"].append(train_ctc_loss)
            train_metrics_history["train_l2_loss"].append(train_l2_loss)
            wandb.log(
                {
                    "train_loss": loss.item(),
                    "train_ctc_loss": train_ctc_loss,
                    "train_l2_loss": train_l2_loss,
                    "samples": epoch * total_train_steps * batch_size
                    + (i + 1) * batch_size,
                }
            )
            pbar.set_postfix({"loss": loss.item()})
            pbar.update(1)


def evaluate_model(
    model,
    dl: data_loader.DataLoader,
    epoch,
    num_epochs,
    total_records,
    batch_size,
    nnx_metrics: nnx.MultiMetric,
):
    # Compute the metrics on the tuning set after each training epoch.
    bar_format = "{desc}[{n_fmt}/{total_fmt}]{postfix} [{elapsed}<{remaining}]"
    model.eval()  # Set model to evaluation model: e.g. use stored batch statistics
    total_steps = total_records // batch_size
    with tqdm.tqdm(
        desc=f"[eval] epoch: {epoch}/{num_epochs}, ",
        total=total_steps,
        bar_format=bar_format,
        leave=True,
    ) as pbar:
        nnx_metrics.reset()  # Reset the eval metrics
        for i in range(total_steps):
            batch = dl.get_batch()
            eval_step(
                model,
                jnp.array(batch["input_seq"]),
                jnp.array(batch["input_paddings"]),
                jnp.array(batch["label"]),
                jnp.array(batch["label_paddings"]),
                nnx_metrics,
            )
            pbar.update(1)

        for metric, value in nnx_metrics.compute().items():
            wandb.log({f"test_{metric}": value})
            eval_metrics_history[f"test_{metric}"].append(value)

    print(f"[eval] epoch: {epoch}/{num_epochs}")
    print(f"- avg loss: {nnx_metrics.compute()['loss']}")
    print(f"- label_error_rate: {nnx_metrics.compute()['label_error_rate']}")


def count_params(model: lstm.Network):
    params = nnx.state(model, nnx.Param)
    total_params = sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(params))

    for path, x in jax.tree_util.tree_leaves_with_path(params):
        print(path, np.prod(x.shape))
    return total_params


def overfit_on_one_example(
    model,
    optimizer,
    train_data_loader,
    l2_reg,
):
    batch = train_data_loader.get_batch()
    for i in range(200):
        loss, aux_info = train_step(
            model,
            optimizer,
            jnp.array(batch["input_seq"]),
            jnp.array(batch["input_paddings"]),
            jnp.array(batch["label"]),
            jnp.array(batch["label_paddings"]),
            l2_reg,
        )
        train_ctc_loss = aux_info["ctc_loss"]
        train_l2_loss = aux_info["l2_loss"]
        print(
            f"iter: {i}, loss: {loss}, ctc_loss: {train_ctc_loss}, l2_loss: {train_l2_loss}"
        )

    loss, label_error_rate = compute_metrics(
        model,
        jnp.array(batch["input_seq"]),
        jnp.array(batch["input_paddings"]),
        jnp.array(batch["label"]),
        jnp.array(batch["label_paddings"]),
    )
    print(f"label_error_rate: {label_error_rate}")
    logits_btv = model(
        jnp.array(batch["input_seq"]),
        jnp.array(batch["input_paddings"]),
    )
    logprobs_btv = nnx.log_softmax(logits_btv)
    best_paths_probs = decode.best_path_decode(
        logprobs_btv, jnp.array(batch["input_paddings"])
    )
    best_path = best_paths_probs[0][0]
    best_path_collapsed = metrics.collapse_repetitions(best_path, 0)
    print("best_path", best_path)
    print("best_path_collapsed", best_path_collapsed)
    print(
        "best_path detok", [LABEL_2_PHN[lbl] if lbl != 0 else "BL" for lbl in best_path]
    )
    print(
        "best_path_collapsed detok",
        [LABEL_2_PHN[lbl] if lbl != 0 else "BL" for lbl in best_path_collapsed],
    )
    print("label", batch["label"])

    def label_detok(label):
        detok = []
        for lbl in label:
            if lbl == 999:
                continue
            else:
                detok.append(LABEL_2_PHN[lbl])
        return detok

    print(
        "label detok",
        label_detok(batch["label"][0].numpy()),
    )


def main():
    args = parser.parse_args()

    wandb.login(key=args.wandb_key)

    with open(args.train_config, "r") as f:
        train_cfg = json.load(f)

    total_train_records = sum([1 for _ in tf.data.TFRecordDataset(args.train_tfr)])
    total_tuning_records = sum([1 for _ in tf.data.TFRecordDataset(args.tune_tfr)])

    train_data_loader = data_loader.DataLoader(
        args.train_tfr,
        train_cfg["input_dim"],
        train_cfg["batch_size"],
        train_cfg["num_epochs"],
    )

    tuning_data_loader = data_loader.DataLoader(
        args.tune_tfr,
        train_cfg["input_dim"],
        train_cfg["batch_size"],
        train_cfg["num_epochs"],
    )

    model = lstm.Network(
        train_cfg["is_bilstm"],
        train_cfg["input_dim"],
        train_cfg["hidden_dim"],
        train_cfg["output_dim"],
        nnx.Rngs(params=0),
    )

    total_params = count_params(model)

    tx = optax.chain(
        optax.clip_by_global_norm(20.0),
        optax.sgd(train_cfg["learning_rate"], momentum=train_cfg["momentum"]),
    )
    optimizer = nnx.Optimizer(model, tx)

    run = wandb.init(
        project="ctc",  # Specify your project
        config={  # Track hyperparameters and metadata
            "optimizer": {
                "name": "sgd with momentum",
                "lr": train_cfg["learning_rate"],
                "momentum": train_cfg["momentum"],
            },
            "model": {
                "input_dim": train_cfg["input_dim"],
                "hidden_dim": train_cfg["hidden_dim"],
                "output_dim": train_cfg["output_dim"],
            },
            "training": {
                "batch_size": train_cfg["batch_size"],
                "epochs": train_cfg["num_epochs"],
                "l2_reg": train_cfg["l2_reg"],
            },
            "feat_prep": "librosa",
            "total_params": total_params,
        },
    )

    wandb.define_metric("train_loss", step_metric="samples")
    wandb.define_metric("test_loss", step_metric="samples")
    wandb.define_metric("test_label_error_rate", step_metric="samples")

    num_epochs = train_cfg["num_epochs"]
    overfit_on_one_example(model, optimizer, train_data_loader, 0.0)
    # for epoch in range(num_epochs):
    #     train_one_epoch(
    #         model,
    #         optimizer,
    #         train_data_loader,
    #         total_train_records,
    #         epoch,
    #         num_epochs,
    #         train_cfg["batch_size"],
    #         train_cfg["l2_reg"],
    #     )
    #     evaluate_model(
    #         model,
    #         tuning_data_loader,
    #         epoch,
    #         num_epochs,
    #         total_tuning_records,
    #         train_cfg["batch_size"],
    #         nnx_metrics=eval_metrics,
    #     )


if __name__ == "__main__":
    main()
