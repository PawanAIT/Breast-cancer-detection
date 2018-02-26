#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Convolutional Neural Network Estimator for MNIST, built with tf.layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)
import re
import numpy
import os
import tensorflow as tf
from random import randint

def read_pgm(filename, byteorder='>'):
  with open(filename, 'rb') as f:
    buffer = f.read()
  try:
    header, width, height, maxval = re.search(
  b"(^P5\s(?:\s*#.*[\r\n])*"
  b"(\d+)\s(?:\s*#.*[\r\n])*"
  b"(\d+)\s(?:\s*#.*[\r\n])*"
  b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
  except AttributeError:
    raise ValueError("Not a raw PGM file: '%s'" % filename)
  return numpy.frombuffer(buffer,
         dtype='u1' if int(maxval) < 256 else byteorder+'u2',
         count=int(width)*int(height),
         offset=len(header)
        ).reshape((int(height)*int(width)))


def import_images(image_dir, num_images):
  # We'll create an array to hold all of the images, each of which is 1024x1024
  images_tensor = numpy.zeros((num_images, 1024*1024))
  i = 0
  for dirName, subdirList, fileList in os.walk(image_dir):
    for fname in fileList:
        if fname.endswith(".pgm"):
          images_tensor[i] = read_pgm(image_dir+fname, byteorder='<')
          i += 1

  # Create a tensor for the labels
  labels_tensor = numpy.zeros(num_images,dtype=np.int32)
  f = open("data.txt", 'r')
  for line in f:

    # The first value in the line is the database ID
    # some values are duplicated so we have to use this as the key
    image_num = int(line.split()[0].replace("mdb", ""))-1

    # The third value in the line is the class of abnormality present
    abnormality = line.split()[2]

    # we are going to built a one-hot vector for each label,
    # using the abnormality of the mammogram
    if abnormality == "CALC":
      labels_tensor[image_num] = 1
    elif abnormality == "CIRC":
      labels_tensor[image_num] = 2
    elif abnormality == "SPIC":
      labels_tensor[image_num] = 3
    elif abnormality == "MISC":
      labels_tensor[image_num] = 4
    elif abnormality == "ARCH":
      labels_tensor[image_num] = 5
    elif abnormality == "ASYM":
      labels_tensor[image_num] = 6
    elif abnormality == "NORM":
      labels_tensor[image_num] = 7

  return images_tensor, labels_tensor


def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # MNIST images are 28x28 pixels, and have one color channel
  input_layer = tf.reshape(features["x"], [-1, 1024, 1024, 1])

  # Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 28, 28, 1]
  # Output Tensor Shape: [batch_size, 28, 28, 32]
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
  #print(conv1.shape) #(10, 1024, 1024, 32)

  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 28, 28, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 32]
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[8, 8], strides=8)
  #print(pool1.shape)
  #(10, 128, 128, 32)

  # Convolutional Layer #2
  # Computes 64 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 14, 14, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 64]
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[11, 11],
      padding="same",
      activation=tf.nn.relu)
  #print(conv2.shape)
  #(10, 128, 128, 64)


  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 14, 14, 64]
  # Output Tensor Shape: [batch_size, 7, 7, 64]
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[8, 8], strides=8)
  #print(pool2.shape)
  #(10, 16, 16, 64)

  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 7, 7, 64]
  # Output Tensor Shape: [batch_size, 7 * 7 * 64]
  pool2_flat = tf.reshape(pool2, [-1, 16 * 16 * 64])

  # Dense Layer
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 7 * 7 * 64]
  # Output Tensor Shape: [batch_size, 1024]
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

  # Add dropout operation; 0.6 probability that element will be kept
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits layer
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, 10]
  logits = tf.layers.dense(inputs=dropout, units=10)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
  # Load training and eval data
  #mnist = tf.contrib.learn.datasets.load_dataset("mnist")
  
  train_data  , train_labels = import_images("./images/", 322)

  eval_data , eval_labels = train_data  , train_labels
  mnist_classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn, model_dir="/tmp/breast_cancer")

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)

  # Train the model
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": train_data.astype(np.float32)},
      y=train_labels.astype(np.int32),
      batch_size=10,
      num_epochs=None,
      shuffle=True)
  mnist_classifier.train(
      input_fn=train_input_fn,
      steps=10,
      hooks=[logging_hook])

  # Evaluate the model and print results
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": eval_data.astype(np.float32)},
      y=eval_labels.astype(np.int32),
      batch_size=5,
      num_epochs=1,
      shuffle=False)
  eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)

if __name__ == "__main__":
  tf.app.run()


#(55000, 784) (55000,)

'''




'''
