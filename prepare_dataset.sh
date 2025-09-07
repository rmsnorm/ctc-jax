# Prepare train dataset
BASE_DIR=/home/apoorv/Projects/datasets/timit/TIMIT/TRAIN
OUT_DIR=/home/apoorv/Projects/datasets/timit/TIMIT/train_processed
bazel run :prepare_timit_dataset -- \
--base_dir=${BASE_DIR} \
--output_dir=${OUT_DIR} \
--mfcc_config=/home/apoorv/Projects/ctc-jax/configs/mfcc_13.json


# Prepare test dataset
BASE_DIR=/home/apoorv/Projects/datasets/timit/TIMIT/TEST
OUT_DIR=/home/apoorv/Projects/datasets/timit/TIMIT/test_processed
bazel run :prepare_timit_dataset -- \
--base_dir=${BASE_DIR} \
--output_dir=${OUT_DIR} \
--mfcc_config=/home/apoorv/Projects/ctc-jax/configs/mfcc_13.json