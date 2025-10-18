set -e
set -x

WANDB_KEY=""
TRAIN_FILE="/home/apoorv/Projects/datasets/timit/TIMIT/train_processed/data.tfrecord"
TEST_FILE="/home/apoorv/Projects/datasets/timit/TIMIT/train_processed/data_tuning.tfrecord"

bazel run :train_bilstm -- \
--train_tfr=${TRAIN_FILE} \
--tune_tfr=${TEST_FILE} \
--train_config=/home/apoorv/Projects/ctc-jax/configs/lstm_reduced_set.json \
--wandb_key=${WANDB_KEY} \
--checkpoint_dir=/home/apoorv/Projects/ctc-checkpoints/
