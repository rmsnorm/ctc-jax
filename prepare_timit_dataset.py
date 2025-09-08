"""Prepare train and test dataset from TIMIT"""

import argparse
import glob
import os
import pickle
from typing import Sequence

import mfcc
import numpy as np
import phoneset
import sphfile
import tensorflow as tf

parser = argparse.ArgumentParser(
    prog="TimitBuilder",
    description="This binary creates pairs of"
    "(X, y) files where X is a sequence of MFCC features and y is the unaligned label sequence",
)
parser.add_argument(
    "--base_dir", help="Base dir containing the TIMIT dataset", type=str
)
parser.add_argument(
    "--output_dir", help="Output dir containing the processed TIMIT dataset", type=str
)
parser.add_argument(
    "--mfcc_config", help="Json file containing the MFCC config", type=str
)
parser.add_argument("--is_train", help="Set to true if computing for train.", type=bool)
parser.add_argument(
    "--train_stats_file",
    help="if is_train, will compute feature statistics. else, will load precomputed stats",
    type=str,
)
parser.add_argument(
    "--add_gaussian_noise",
    help="if is_train, will add gaussian noise with std dev 0.6.",
    type=bool,
)


# reserve 0 for CTC blank symbol.
PHN_2_LABEL = dict(zip(phoneset.PHONE_SET, range(1, len(phoneset.PHONE_SET) + 1)))
LABEL_2_PHN = dict([(v, k) for k, v in PHN_2_LABEL.items()])


def parse_phn_file(phn_file_path) -> Sequence[int]:
    """Extract phoneme sequence given the phn file"""
    phn_seq = []
    with open(phn_file_path, "r", encoding="ascii") as f:
        lines = f.read().splitlines()
        for line in lines:
            _, _, phoneme = line.split(" ")
            phn_seq.append(PHN_2_LABEL[phoneme])

    return phn_seq


def pad_feat(feat: np.ndarray, max_len: int) -> np.ndarray:
    """Pad input sequence of features to max length sequence"""
    return np.pad(
        feat,
        pad_width=((0, max_len - feat.shape[0]), (0, 0)),
        mode="constant",
        constant_values=0.0,
    )


# The following functions can be used to convert a value to a type compatible
# with tf.train.Example.
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    if isinstance(value, list):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    if isinstance(value, list):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def create_tf_dataset(
    wav_files: Sequence[str],
    out_dir: str,
    mfcc_computer: mfcc.MfccComputer,
    is_train: bool,
    train_stats_file: str,
    add_gaussian_noise: bool,
):
    max_feat_len = 0
    max_phn_len = 0
    path_keys = [x.strip(".WAV") for x in wav_files]

    dataset = {}

    for path_key in path_keys:
        wav_path = path_key + ".WAV"
        phn_path = path_key + ".PHN"

        sf = sphfile.SPHFile(wav_path)
        feat = mfcc_computer(sf.content)
        feat_len = feat.shape[0]
        max_feat_len = max(max_feat_len, feat_len)

        label_seq = parse_phn_file(phn_path)
        label_len = len(label_seq)
        max_phn_len = max(max_phn_len, label_len)

        dataset[path_key] = {
            "feat": feat,
            "feat_len": feat_len,
            "label_seq": label_seq,
            "label_len": label_len,
        }

    if is_train:
        feats = []
        for path_key in path_keys:
            feat_btd = dataset[path_key]["feat"]
            feat_flattened_d = feat_btd.reshape((-1, feat_btd.shape[-1]))
            feats.extend(feat_flattened_d)
        feats = np.array(feats)

        print("all_feats shape", feats.shape)

        feats_mu = np.mean(feats, axis=0, keepdims=True)
        feats_std = np.std(feats, axis=0, keepdims=True)

        print("feats_mu shape", feats_mu.shape)

        with open(train_stats_file, "wb") as f:
            pickle.dump(feats_mu, f)
            pickle.dump(feats_std, f)
    else:
        with open(train_stats_file, "rb") as f:
            feats_mu = pickle.load(f)
            feats_std = pickle.load(f)

    for path_key in path_keys:
        feat_normed = (dataset[path_key]["feat"] - feats_mu) / feats_std
        if is_train and add_gaussian_noise:
            feat_normed += np.random.normal(scale=0.6, size=feat_normed.shape)
        dataset[path_key]["feat_normed"] = feat_normed

    writer = tf.io.TFRecordWriter(os.path.join(out_dir, "data.tfrecord"))

    for i, path_key in enumerate(path_keys):
        feat_len = dataset[path_key]["feat_len"]
        padded_feat = pad_feat(dataset[path_key]["feat_normed"], max_feat_len)
        label_len = dataset[path_key]["label_len"]
        padded_label_seq = dataset[path_key]["label_seq"] + [999] * (
            max_phn_len - label_len
        )

        feature = {
            "input_seq": _bytes_feature(padded_feat.astype(np.float32).tobytes()),
            "input_paddings": _float_feature(
                [0.0] * feat_len + [1.0] * (max_feat_len - feat_len)
            ),
            "label": _int64_feature(padded_label_seq),
            "label_paddings": _float_feature(
                [0.0] * label_len + [1.0] * (max_phn_len - label_len)
            ),
        }

        tf_example = tf.train.Example(
            features=tf.train.Features(feature=feature)
        ).SerializeToString()
        writer.write(tf_example)
        if i % 100 == 0:
            print(f"Created tf_examples for {i} files")


def main():
    args = parser.parse_args()
    mfcc_config = mfcc.MfccConfig.from_json(args.mfcc_config)
    mfcc_computer = mfcc.MfccComputer(mfcc_config)
    base_dir = args.base_dir
    output_dir = args.output_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    wav_files = glob.glob("**/*.WAV", root_dir=base_dir, recursive=True)
    wav_files = [os.path.join(base_dir, wav_file) for wav_file in wav_files]

    create_tf_dataset(
        wav_files,
        args.output_dir,
        mfcc_computer,
        args.is_train,
        args.train_stats_file,
        args.add_gaussian_noise,
    )


if __name__ == "__main__":
    main()
