from __future__ import print_function

import tensorflow as tf

# Simple hello world using Tensorflow

# Create a Constant op
# The op is added as a node to the defaulr graph.
#
# The calue returned by the constructor represents the output
# of the Constant op.
hello = tf.constant('Hello, Tensorflow')

# Start tf session
sess = tf.Session()

# Run the op
print(sess.run(hello))