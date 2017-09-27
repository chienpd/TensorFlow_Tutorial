from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.tensor_forest.python import tensor_forest

# ignore all GPUs, tf random forest does not benefit from it
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../data/", one_hot=False)

# Parameters
num_steps = 500     # Total srep train
batch_size = 1024   # The number of samples per batch
num_classes = 10    # The 10 digist
num_features = 784  # Each image is 28*28 pixels
num_trees = 10
max_nodes = 1000

# Input and Target data
X = tf.placeholder(tf.float32, shape=[None, num_features])
# For random forest, labels must be integers (the clas id)
Y = tf.placeholder(tf.int32, shape=[None])
