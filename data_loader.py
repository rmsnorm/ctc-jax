"""Dataset loader for TIMIT data."""

import tensorflow as tf


class DataLoader:
    def __init__(
        self,
        tfrecord_file,
        mfcc_feat_dim,
        batch_size,
        num_epochs=1,
        buffer_size=65536,
        shuffle_buffer_size=100,
    ):
        self.tfrecord_file = tfrecord_file
        self.mfcc_feat_dim = mfcc_feat_dim
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.dataset = tf.data.TFRecordDataset(tfrecord_file, buffer_size=buffer_size)
        self.dataset = (
            self.dataset.map(self._parse_example_fn)
            .repeat(num_epochs)
            .shuffle(buffer_size=shuffle_buffer_size, reshuffle_each_iteration=True)
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

        self.iterator = iter(self.dataset)

    def _parse_example_fn(self, example):
        """Parse a single TFRecord example."""
        feature_description = {
            "input_seq": tf.io.FixedLenFeature([], tf.string),
            "input_paddings": tf.io.VarLenFeature(tf.float32),
            "label": tf.io.VarLenFeature(tf.int64),
            "label_paddings": tf.io.VarLenFeature(tf.float32),
        }

        parsed_example = tf.io.parse_single_example(example, feature_description)

        input_seq = tf.io.decode_raw(parsed_example["input_seq"], tf.float32)
        input_seq = tf.reshape(input_seq, [-1, self.mfcc_feat_dim])

        return {
            "input_seq": input_seq,
            "input_paddings": tf.sparse.to_dense(parsed_example["input_paddings"]),
            "label": tf.sparse.to_dense(parsed_example["label"]),
            "label_paddings": tf.sparse.to_dense(parsed_example["label_paddings"]),
        }

    def get_batch(self):
        return next(self.iterator)
