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
import os
import cv2 as cv

tf.logging.set_verbosity(tf.logging.INFO)


def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  input_layer = tf.reshape(features["x"], [-1, 64, 64, 1])

  # Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 64px, 64px, 1]
  # Output Tensor Shape: [batch_size, 64px, 64px, 32]
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[10, 10],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 64px, 64px, 32]
  # Output Tensor Shape: [batch_size, 32px, 32px, 32]
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2
  # Computes 64 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 32px, 32px, 32]
  # Output Tensor Shape: [batch_size, 32px, 32px, 64]
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 32px, 32px, 64]
  # Output Tensor Shape: [batch_size, 16, 16, 64]
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 16, 16, 64]
  # Output Tensor Shape: [batch_size, 16 * 16 * 64]
  pool2_flat = tf.reshape(pool2, [-1, 16 * 16 * 64])

  # Dense Layer
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 16 * 16 * 64]
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








def dataset_input_fn(filenames):
  # filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
  dataset = tf.data.TFRecordDataset(filenames)

  # Use `tf.parse_single_example()` to extract data from a `tf.Example`
  # protocol buffer, and perform any additional per-record preprocessing.
  def parser(record):
    keys_to_features = {
        "image_data": tf.FixedLenFeature((), tf.string, default_value=""),
        "date_time": tf.FixedLenFeature((), tf.int64, default_value=""),
        "label": tf.FixedLenFeature((), tf.int64,
                                    default_value=tf.zeros([], dtype=tf.int64)),
    }
    parsed = tf.parse_single_example(record, keys_to_features)

    # Perform additional preprocessing on the parsed data.
    image = tf.image.decode_jpeg(parsed["image_data"])
    image = tf.reshape(image, [299, 299, 1])
    label = tf.cast(parsed["label"], tf.int32)

    return {"image_data": image, "date_time": parsed["date_time"]}, label

  # Use `Dataset.map()` to build a pair of a feature dictionary and a label
  # tensor for each example.
  dataset = dataset.map(parser)
  dataset = dataset.shuffle(buffer_size=10000)
  dataset = dataset.batch(32)
  dataset = dataset.repeat(num_epochs)
  iterator = dataset.make_one_shot_iterator()

  # `features` is a dictionary in which each value is a batch of values for
  # that feature; `labels` is a batch of labels.
  features, labels = iterator.get_next()
  return features, labels










def _parse_function(filename, label):
  print(filename)
  image_string = tf.read_file(filename)
  image_decoded = tf.image.decode_image(image_string)
  # image_resized = tf.image.resize_images(image_decoded, [64, 64])
  return image_decoded, label

def get_sliced_data(evalu):
    base_dir = '/dataset/processed/masked/'
    filenames = []
    labels = []
    raw_binary = []
    for dirname in os.listdir(base_dir):
        dirnum = int(dirname)
        for filename in os.listdir(base_dir + dirname):
            if (filename.startswith('00000') and evalu) or (not filename.startswith('00000') and not evalu):
                filenames.append( base_dir + dirname + '/' + filename)
                labels.append( dirnum )
                img = cv.imread(filenames[-1])
                img_data = []
                for i in range(0, 64):
                    new_row = []
                    for j in range(0, 64):
                        if img[i][j][0] > 100:
                            new_row.append(1)
                        else: 
                            new_row.append(0)
                            
                    img_data.append(new_row)
                raw_binary.append(img_data)



    print(str(len(raw_binary)))
    print(str(len(labels)))
    # return tf.data.Dataset.from_tensor_slices((raw_binary, labels))
    return (tf.convert_to_tensor(raw_binary,dtype=tf.float16), np.array(labels))




def main(unused_argv):
  # Load training and eval data
  #mnist = tf.contrib.learn.datasets.load_dataset("mnist")
  #train_data = mnist.train.images  # Returns np.array
  #train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
  #eval_data = mnist.test.images  # Returns np.array
  #eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
  
  train_data, train_labels = get_sliced_data(False)

  #dataset = get_sliced_data(False)
  ##dataset = dataset.map(_parse_function)
  #train_data = dataset.images
  #train_labels = dataset.labels
  
  #eval_data = dataset.test.images
  #eval_labels = dataset.test.labels
  
  eval_data, eval_labels = get_sliced_data(True)
  
  #evalu = get_sliced_data(True)
  ##evalu = evalu.map(_parse_function)
  #eval_data = evalu.images
  #eval_labels = evalu.labels
  
  # Create the Estimator
  mnist_classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn, model_dir="/tmp/hands")

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)











train_spec = tf.estimator.TrainSpec(input_fn=lambda: imgs_input_fn(test_files,
                                                                   labels=test_labels,
                                                                   perform_shuffle=True,
                                                                   repeat_count=5,
                                                                   batch_size=20), 
                                    max_steps=500)
eval_spec = tf.estimator.EvalSpec(input_fn=lambda: imgs_input_fn(test_files,
                                                                 labels=test_labels,
                                                                 perform_shuffle=False,
                                                                 batch_size=1))

tf.estimator.train_and_evaluate(est_c`atvsdog, train_spec, eval_spec)










  # Train the model
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": train_data},
      y=train_labels,
      batch_size=100,
      num_epochs=None,
      shuffle=True)


  mnist_classifier.train(
      input_fn=train_input_fn,
      steps=20000,
      hooks=[logging_hook])



  # Evaluate the model and print results
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": eval_data},
      y=eval_labels,
      num_epochs=1,
      shuffle=False)

  eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)


if __name__ == "__main__":
  tf.app.run()
