# Python

import tensorflow as tf
import os
import numpy as np
import cv2 as cv


def get_sliced_data(evalu):
    base_dir = '/dataset/processed/masked/'
    filenames = []
    labels = []
    raw_binary = []
    tensors = []
    for dirname in os.listdir(base_dir):
        dirnum = int(dirname)
        count = 0
        for filename in os.listdir(base_dir + dirname):
            count += 1
            if (count <= 10 and evalu) or (count > 10 and not evalu):
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
    print tensors.shape
    return (tensors, np.array(labels))





feature_columns = [tf.feature_column.numeric_column('x', shape=[64, 64])]

classifier = tf.estimator.DNNClassifier(
 feature_columns=feature_columns,
 hidden_units=[256, 32],
 optimizer=tf.train.AdamOptimizer(1e-4),
 n_classes=3,
 dropout=0.1,
 model_dir="./tmp/hand_model_5"
)


training_data = get_sliced_data(False)


def _parse_me(superimg):
    t = tf.convert_to_tensor(superimg,dtype=tf.float16)
    tf.reshape(t, [-1])


train_input_fn = tf.estimator.inputs.numpy_input_fn(
 x={"x": training_data[0]},
 y=training_data[1],
 num_epochs=None,
 batch_size=500,
 shuffle=True
)

#classifier.train(input_fn=train_input_fn, steps=100000)

test_data, test_labels = get_sliced_data(True)

# Define the test inputs
test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": test_data},
    y=test_labels,
    num_epochs=1,
    shuffle=False
)

# Evaluate accuracy
accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]
print("\nTest Accuracy: {0:f}%\n".format(accuracy_score*100))



#ds_predict_tf = classifier.predict(input_fn = test_input_fn)

#print(list(ds_predict_tf))

