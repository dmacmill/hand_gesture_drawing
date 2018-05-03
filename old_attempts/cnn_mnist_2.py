# Python

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
   #loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
  loss = tf.losses.mean_squared_error(
    labels,
    predictions,
    weights=1.0,
    scope=None,
    loss_collection=tf.GraphKeys.LOSSES,
    reduction=Reduction.SUM_BY_NONZERO_WEIGHTS
  )

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




def get_sliced_data(evalu):
    base_dir = '/dataset/processed/masked/'
    filenames = []
    labels = []
    raw_binary = []
    tensors = []
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
                #t = tf.convert_to_tensor(img_data,dtype=tf.float16)
                t = np.array(img_data, dtype=np.float16)
                #t = tf.reshape(t, [-1])
                t = t.flatten()
                tensors.append(t)
    
    tensors = np.array(tensors)
    return (tensors, np.array(labels))








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





feature_columns = [tf.feature_column.numeric_column('x', shape=[64, 64])]



training_data = get_sliced_data(False)


def _parse_me(superimg):
    t = tf.convert_to_tensor(superimg,dtype=tf.float16)
    tf.reshape(t, [-1])


train_input_fn = tf.estimator.inputs.numpy_input_fn(
 x={"x": training_data[0]},
 y=training_data[1],
 num_epochs=None,
 batch_size=50,
 shuffle=True
)

test_data, test_labels = get_sliced_data(True)

# Define the test inputs
test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": test_data},
    y=test_labels,
    num_epochs=1,
    shuffle=False
)




mnist_classifier.train(
    input_fn=train_input_fn,
    steps=20000,
    hooks=[logging_hook])



eval_results = mnist_classifier.evaluate(input_fn=test_input_fn)
print("\nTest Accuracy: {0:f}%\n".format(eval_results*100))


if __name__ == "__main__":
  tf.app.run()


