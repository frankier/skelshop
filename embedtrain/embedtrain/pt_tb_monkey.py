# See https://github.com/pytorch/pytorch/issues/30966
import tensorboard as tb
import tensorflow as tf

tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
