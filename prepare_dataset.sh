# Prepare train dataset
BASE_DIR=/home/apoorv/Projects/datasets/timit/TIMIT/TRAIN
OUT_DIR=/home/apoorv/Projects/datasets/timit/TIMIT/train_processed
bazel run :prepare_timit_dataset -- \
--base_dir=${BASE_DIR} \
--output_dir=${OUT_DIR} \
--mfcc_config=/home/apoorv/Projects/ctc-jax/configs/mfcc_39.json \
--is_train=true \
--train_stats_file=${OUT_DIR}/feats_stats.pkl \
--add_gaussian_noise=true \
--use_reduced_phone_set=true


# Prepare test dataset
BASE_DIR=/home/apoorv/Projects/datasets/timit/TIMIT/TEST
TEST_OUT_DIR=/home/apoorv/Projects/datasets/timit/TIMIT/test_processed
bazel run :prepare_timit_dataset -- \
--base_dir=${BASE_DIR} \
--output_dir=${TEST_OUT_DIR} \
--mfcc_config=/home/apoorv/Projects/ctc-jax/configs/mfcc_39.json \
--train_stats_file=${OUT_DIR}/feats_stats.pkl \
--use_reduced_phone_set=true