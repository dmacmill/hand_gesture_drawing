# Python

print ( ' here comes the data .. . . . ')


import os
import tensorflow as tf


import numpy as np







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




























def imgs_input_fn(filenames, labels=None, perform_shuffle=False, repeat_count=1, batch_size=1):
    def _parse_function(filename, label):
        input_name = 'img_data'
        image_string = tf.read_file(filename)
        image = tf.image.decode_image(image_string, channels=3)
        image.set_shape([None, None, None])
        image = tf.image.resize_images(image, [150, 150])
        image = tf.subtract(image, 116.779) # Zero-center by mean pixel
        image.set_shape([150, 150, 3])
        image = tf.reverse(image, axis=[2]) # 'RGB'->'BGR'
        d = dict(zip([input_name], [image])), label
        return d
    if labels is None:
        labels = [0]*len(filenames)
    labels=np.array(labels)
    # Expand the shape of "labels" if necessory
    if len(labels.shape) == 1:
        labels = np.expand_dims(labels, axis=1)
    filenames = tf.constant(filenames)
    labels = tf.constant(labels)
    labels = tf.cast(labels, tf.float32)
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(_parse_function)
    if perform_shuffle:
        # Randomizes input using a window of 256 elements (read into memory)
        dataset = dataset.shuffle(buffer_size=256)
    dataset = dataset.repeat(repeat_count)  # Repeats dataset this # times
    dataset = dataset.batch(batch_size)  # Batch size to use
    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels




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
                #img = cv.imread(filenames[-1])
                #img_data = []
                #for i in range(0, 64):
                #    new_row = []
                #    for j in range(0, 64):
                #        if img[i][j][0] > 100:
                #            new_row.append(1)
                #        else: 
                #            new_row.append(0)
                            
                #    img_data.append(new_row)
                #raw_binary.append(img_data)



    print(str(len(raw_binary)))
    print(str(len(labels)))
    
    return (filenames, labels)
    # return tf.data.Dataset.from_tensor_slices((raw_binary, labels))
    #return (tf.convert_to_tensor(raw_binary,dtype=tf.float16), np.array(labels))







# Reads an image from a file, decodes it into a dense tensor, and resizes it
# to a fixed shape.
def _parse_function(filename, label):
  image_string = tf.read_file(filename)
  image_decoded = tf.image.decode_image(image_string)
  image_resized = tf.image.resize_images(image_decoded, [64, 64])
  return image_resized, label








# Create the Estimator
mnist_classifier = tf.estimator.Estimator(
    model_fn=cnn_model_fn, model_dir="/tmp/hands")

tensors_to_log = {"probabilities": "softmax_tensor"}
logging_hook = tf.train.LoggingTensorHook(
    tensors=tensors_to_log, every_n_iter=50)

filenames, labels = get_sliced_data(False)



# Train the model
train_input_fn = imgs_input_fn(filenames, labels=labels, perform_shuffle=True, repeat_count=5, batch_size=200)


mnist_classifier.train(
    input_fn=train_input_fn,
    steps=20000,
    hooks=[logging_hook])



# Evaluate the model and print results
eval_input_fn = imgs_input_fn(filenames, labels=labels, perform_shuffle=False, repeat_count=1, batch_size=1)


eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
print(eval_results)


tf.app.run()



#dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
#dataset = dataset.map(_parse_function)

