"""Prepare train and test dataset from TIMIT"""

import argparse
import glob
from typing import Sequence
import os
import numpy as np
import tensorflow as tf

import mfcc
import phoneset
import sphfile

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

# reserve 0 for CTC blank symbol and 1 for padding symbol.
PHN_2_LABEL = dict(zip(phoneset.PHONE_SET, range(2, len(phoneset.PHONE_SET) + 2)))
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
    wav_files: Sequence[str], mfcc_computer: mfcc.MfccComputer, out_dir: str
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

    writer = tf.io.TFRecordWriter(os.path.join(out_dir, "data.tfrecord"))

    for i, path_key in enumerate(path_keys):
        feat_len = dataset[path_key]["feat_len"]
        padded_feat = pad_feat(dataset[path_key]["feat"], max_feat_len)
        label_len = dataset[path_key]["label_len"]
        padded_label_seq = dataset[path_key]["label_seq"] + [1] * (
            max_phn_len - label_len
        )

        feature = {
            "input_seq": _bytes_feature(padded_feat.tobytes()),
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

    create_tf_dataset(wav_files, mfcc_computer, args.output_dir)


if __name__ == "__main__":
    main()
