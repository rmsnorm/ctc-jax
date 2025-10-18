"""Train BiLSTM with CTC loss on prepared dataset."""

import argparse
import json
import data_loader
import lstm
import metrics
import decode
import tqdm
from flax import nnx
from flax import traverse_util
import optax
import jax.numpy as jnp
import tensorflow as tf
import wandb
import numpy as np
import jax
import phoneset
import os
import orbax.checkpoint as ocp

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
parser.add_argument("--checkpoint_dir", help="Path to store checkpoints", type=str)

train_metrics = nnx.MultiMetric(
    ctc_loss=nnx.metrics.Average("ctc_loss"),
    l2_loss=nnx.metrics.Average("l2_loss"),
    total_loss=nnx.metrics.Average("total_loss"),
)

eval_loss_metrics = nnx.MultiMetric(ctc_loss=nnx.metrics.Average("ctc_loss"))
eval_ler_metrics = nnx.MultiMetric(
    label_error_rate=nnx.metrics.Average("label_error_rate")
)


@nnx.jit
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
            l2_loss += jnp.sum(param**2)

    total_loss = ctc_loss + l2_reg * l2_loss

    # Auxiliary information (not used for gradients)
    aux_info = {
        "ctc_loss": ctc_loss,
        "l2_loss": l2_loss,
        "total_loss": total_loss,
    }

    return total_loss, aux_info


# @nnx.jit
def compute_eval_loss(
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
    return loss


def compute_ler_batch(
    model: lstm.Network, input_btd, logit_paddings, label, label_paddings
):
    bsz = input_btd.shape[0]
    label_errors = 0
    logits_btv = model(input_btd, logit_paddings)
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

    return label_errors


@nnx.jit
def train_step(
    model, optimizer, input_btd, logit_paddings, label, label_paddings, l2_reg
):
    grad_fn = nnx.value_and_grad(compute_loss, has_aux=True)
    (loss, aux_info), grads = grad_fn(
        model, input_btd, logit_paddings, label, label_paddings, l2_reg
    )

    norm_fn = lambda g: jnp.linalg.norm(g)

    grad_norms = jax.tree_util.tree_map_with_path(
        lambda path, leaf: norm_fn(leaf), grads.to_pure_dict()
    )
    flattened_norms = traverse_util.flatten_dict(grad_norms)
    formatted_norms = {
        "grad_norm." + ".".join(map(str, path)): value
        for path, value in flattened_norms.items()
    }

    optimizer.update(grads, value=aux_info["ctc_loss"])
    return loss, aux_info, formatted_norms


def eval_step(model, input_btd, logit_paddings, label, label_paddings, nnx_metrics):
    loss = compute_eval_loss(model, input_btd, logit_paddings, label, label_paddings)
    nnx_metrics.update(ctc_loss=loss)


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
            loss, aux_info, formatted_norms = train_step(
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
            train_metrics.update(
                ctc_loss=train_ctc_loss, l2_loss=train_l2_loss, total_loss=loss
            )

            current_lr = optimizer.opt_state.hyperparams["learning_rate"]
            # current_lr = float(optimizer.opt_state.inner_state[2].scale) * 0.01
            d = {
                "train_loss": loss.item(),
                "train_ctc_loss": train_ctc_loss,
                "train_l2_loss": train_l2_loss,
                "lr": current_lr,
                "samples": epoch * total_train_steps * batch_size
                + (i + 1) * batch_size,
            }
            d.update(formatted_norms)
            wandb.log(d)
            pbar.set_postfix({"loss": loss.item()})
            pbar.update(1)

    print(f"[train] epoch: {epoch}/{num_epochs}")
    print(f"- avg ctc loss: {train_metrics.compute()['ctc_loss']}")
    print(f"- avg l2 loss: {train_metrics.compute()['l2_loss']}")
    print(f"- avg total loss: {train_metrics.compute()['total_loss']}")


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
    model.eval()
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

    print(f"[eval] epoch: {epoch}/{num_epochs}")
    print(f"- avg ctc loss: {nnx_metrics.compute()['ctc_loss']}")


def evaluate_model_ler(
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
    model.eval()
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
            label_errors = compute_ler_batch(
                model,
                jnp.array(batch["input_seq"]),
                jnp.array(batch["input_paddings"]),
                jnp.array(batch["label"]),
                jnp.array(batch["label_paddings"]),
            )
            pbar.update(1)
            nnx_metrics.update(label_error_rate=label_errors)

        for metric, value in nnx_metrics.compute().items():
            wandb.log({f"test_{metric}": value})

    print(f"[eval] epoch: {epoch}/{num_epochs}")
    print(f"- ler: {nnx_metrics.compute()['label_error_rate']}")


def count_params(model: lstm.Network):
    """Count total params in the model."""
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
        32,
        train_cfg["num_epochs"],
    )

    tuning_data_loader_for_ler = data_loader.DataLoader(
        args.tune_tfr,
        train_cfg["input_dim"],
        1,
        train_cfg["num_epochs"],
    )

    model = lstm.Network(
        train_cfg["is_bilstm"],
        train_cfg["input_dim"],
        train_cfg["hidden_dim"],
        train_cfg["output_dim"],
        train_cfg["peephole_connection"],
        nnx.Rngs(params=0),
    )

    total_params = count_params(model)
    print(f"total_params: {total_params}")

    lr_decay_steps = (
        train_cfg["num_epochs"] * total_train_records / train_cfg["batch_size"]
    )
    lr_scheduler = optax.warmup_cosine_decay_schedule(
        init_value=train_cfg["lr_init_value"],
        peak_value=train_cfg["lr_peak_value"],
        warmup_steps=train_cfg["lr_warmup_steps"],
        decay_steps=lr_decay_steps,
    )
    # lr_scheduler = optax.contrib.reduce_on_plateau(
    #     factor=0.5,
    #     patience=5,
    #     rtol=0.1,
    #     accumulation_size=10,
    #     min_scale=1e-7,
    #     cooldown=5,
    # )

    grad_clip = train_cfg["grad_clip"]
    momentum = train_cfg["momentum"]

    @optax.inject_hyperparams
    def create_optimizer(learning_rate, momentum, grad_clip):
        return optax.chain(
            optax.clip_by_global_norm(grad_clip),
            optax.rmsprop(learning_rate=learning_rate, momentum=momentum),
            # learning_rate,
        )

    tx = create_optimizer(lr_scheduler, momentum, grad_clip)
    optimizer = nnx.Optimizer(model, tx)

    run = wandb.init(
        project="ctc",
        config={
            "optimizer": {
                "name": "rmsprop",
                "lr_init_value": train_cfg["lr_init_value"],
                "lr_peak_value": train_cfg["lr_peak_value"],
                "lr_warmup_steps": train_cfg["lr_warmup_steps"],
                "lr_decay_steps": lr_decay_steps,
                "grad_clip": grad_clip,
                "momentum": momentum,
            },
            "model": {
                "input_dim": train_cfg["input_dim"],
                "hidden_dim": train_cfg["hidden_dim"],
                "output_dim": train_cfg["output_dim"],
                "is_bilstm": train_cfg["is_bilstm"],
            },
            "training": {
                "batch_size": train_cfg["batch_size"],
                "epochs": train_cfg["num_epochs"],
                "l2_reg": train_cfg["l2_reg"],
                "grad_clip": grad_clip,
            },
            "feat_prep": "librosa",
            "total_params": total_params,
            "label_set": "reduced_phone_set",
        },
    )

    checkpoint_dir = os.path.join(args.checkpoint_dir, run.name)
    checkpoint_manager = ocp.CheckpointManager(
        ocp.test_utils.erase_and_create_empty(checkpoint_dir),
        options=ocp.CheckpointManagerOptions(
            max_to_keep=20,
            save_interval_steps=5,
            keep_checkpoints_without_metrics=False,
            create=True,
            enable_async_checkpointing=False,
        ),
    )

    wandb.define_metric("train_loss", step_metric="samples")
    wandb.define_metric("test_loss", step_metric="samples")
    wandb.define_metric("test_label_error_rate", step_metric="samples")

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
            train_cfg["l2_reg"],
        )
        evaluate_model(
            model,
            tuning_data_loader,
            epoch,
            num_epochs,
            total_tuning_records,
            32,
            nnx_metrics=eval_loss_metrics,
        )
        evaluate_model_ler(
            model,
            tuning_data_loader_for_ler,
            epoch,
            num_epochs,
            total_tuning_records,
            1,
            nnx_metrics=eval_ler_metrics,
        )

        if epoch % 5 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, str(epoch))
            checkpoint_manager.save(
                epoch,
                args=ocp.args.Composite(state=ocp.args.PyTreeSave(nnx.state(model))),
            )
            wandb.log_artifact(
                checkpoint_path, name=f"checkpoint-{epoch:04}", type="model"
            )

    checkpoint_manager.close()


if __name__ == "__main__":
    main()
