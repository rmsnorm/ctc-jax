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

parser = argparse.ArgumentParser(
    prog="TrainBiLSTM",
    description="This binary trains a BiLSTM with CTC loss on a dataset",
)

parser.add_argument(
    "--train_tfr", help="Path of prepared train dataset in TFRecord format", type=str
)
parser.add_argument(
    "--test_tfr", help="Path of prepared test dataset in TFRecord format", type=str
)
parser.add_argument("--train_config", help="Path of training config.json", type=str)
parser.add_argument("--wandb_key", help="Wandb API key", type=str)


eval_metrics = nnx.MultiMetric(
    loss=nnx.metrics.Average("loss"),
    label_error_rate=nnx.metrics.Average("label_error_rate"),
)

train_metrics_history = {
    "train_loss": [],
}

eval_metrics_history = {"test_loss": [], "test_label_error_rate": []}


def compute_loss(model: lstm.BiLSTM, input_btd, logit_paddings, label, label_paddings):
    """Computes ctc loss."""
    logits_btv = model(input_btd)
    return jnp.mean(
        optax.losses.ctc_loss(
            logits_btv,
            logit_paddings,
            label,
            label_paddings,
        )
    )


def compute_metrics(
    model: lstm.BiLSTM, input_btd, logit_paddings, label, label_paddings
):
    logits_btv = model(input_btd)
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
def train_step(model, optimizer, input_btd, logit_paddings, label, label_paddings):
    grad_fn = nnx.value_and_grad(compute_loss)
    loss, grads = grad_fn(
        model,
        input_btd,
        logit_paddings,
        label,
        label_paddings,
    )
    optimizer.update(grads)
    return loss


def eval_step(model, input_btd, logit_paddings, label, label_paddings, eval_metrics):
    loss, label_error_rate = compute_metrics(
        model, input_btd, logit_paddings, label, label_paddings
    )
    eval_metrics.update(loss=loss, label_error_rate=label_error_rate)


def train_one_epoch(
    model,
    optimizer,
    train_data_loader,
    total_train_records,
    epoch,
    num_epochs,
    batch_size,
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
            loss = train_step(
                model,
                optimizer,
                jnp.array(batch["input_seq"]),
                jnp.array(batch["input_paddings"]),
                jnp.array(batch["label"]),
                jnp.array(batch["label_paddings"]),
            )
            train_metrics_history["train_loss"].append(loss.item())
            # wandb.log(
            #     {
            #         "train_loss": loss.item(),
            #         "samples": epoch * total_train_steps * batch_size
            #         + (i + 1) * batch_size,
            #     }
            # )
            pbar.set_postfix({"loss": loss.item()})
            pbar.update(1)


def evaluate_model(
    model, val_data_loader, epoch, num_epochs, total_val_records, batch_size
):
    # Compute the metrics on the train and val sets after each training epoch.
    bar_format = "{desc}[{n_fmt}/{total_fmt}]{postfix} [{elapsed}<{remaining}]"
    model.eval()  # Set model to evaluation model: e.g. use stored batch statistics
    total_val_steps = total_val_records // batch_size
    with tqdm.tqdm(
        desc=f"[eval] epoch: {epoch}/{num_epochs}, ",
        total=total_val_steps,
        bar_format=bar_format,
        leave=True,
    ) as pbar:
        eval_metrics.reset()  # Reset the eval metrics
        for i in range(total_val_steps):
            batch = val_data_loader.get_batch()
            eval_step(
                model,
                jnp.array(batch["input_seq"]),
                jnp.array(batch["input_paddings"]),
                jnp.array(batch["label"]),
                jnp.array(batch["label_paddings"]),
                eval_metrics,
            )
            pbar.update(1)

        for metric, value in eval_metrics.compute().items():
            # wandb.log({f"test_{metric}": value})
            eval_metrics_history[f"test_{metric}"].append(value)

    print(f"[eval] epoch: {epoch}/{num_epochs}")
    print(f"- total loss: {eval_metrics_history['test_loss'][-1]:0.4f}")
    print(
        f"- label_error_rate: {eval_metrics_history['test_label_error_rate'][-1]:0.4f}"
    )


def count_params(model: lstm.Network):
    params = nnx.state(model, nnx.Param)
    total_params = sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(params))

    for path, x in jax.tree_util.tree_leaves_with_path(params):
        print(path, np.prod(x.shape))
    return total_params


def main():
    args = parser.parse_args()

    wandb.login(key=args.wandb_key)

    with open(args.train_config, "r") as f:
        train_cfg = json.load(f)

    total_train_records = sum([1 for _ in tf.data.TFRecordDataset(args.train_tfr)])
    total_val_records = sum([1 for _ in tf.data.TFRecordDataset(args.test_tfr)])

    train_data_loader = data_loader.DataLoader(
        args.train_tfr,
        train_cfg["input_dim"],
        train_cfg["batch_size"],
        train_cfg["num_epochs"],
    )

    val_data_loader = data_loader.DataLoader(
        args.test_tfr,
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

    optimizer = nnx.Optimizer(
        model, optax.sgd(train_cfg["learning_rate"], momentum=train_cfg["momentum"])
    )

    # run = wandb.init(
    #     project="ctc",  # Specify your project
    #     config={  # Track hyperparameters and metadata
    #         "optimizer": {
    #             "name": "sgd with momentum",
    #             "lr": train_cfg["learning_rate"],
    #             "momentum": train_cfg["momentum"],
    #         },
    #         "model": {
    #             "input_dim": train_cfg["input_dim"],
    #             "hidden_dim": train_cfg["hidden_dim"],
    #             "output_dim": train_cfg["output_dim"],
    #         },
    #         "training": {
    #             "batch_size": train_cfg["batch_size"],
    #             "epochs": train_cfg["num_epochs"],
    #         },
    #         "feat_prep": "librosa",
    #         "total_params": total_params,
    #     },
    # )

    # wandb.define_metric("train_loss", step_metric="samples")
    # wandb.define_metric("test_loss", step_metric="samples")
    # wandb.define_metric("test_label_error_rate", step_metric="samples")

    num_epochs = train_cfg["num_epochs"]
    for epoch in range(num_epochs):
        train_one_epoch(
            model,
            optimizer,
            train_data_loader,
            total_train_records,
            epoch,
            num_epochs,
            train_cfg["batch_size"],
        )
        evaluate_model(
            model,
            val_data_loader,
            epoch,
            num_epochs,
            total_val_records,
            128,
        )


if __name__ == "__main__":
    main()
