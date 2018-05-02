
import tensorflow as tf
import numpy as np


feature_columns = [tf.feature_column.numeric_column('x', shape=[64, 64])]


classifier = tf.estimator.DNNClassifier(
 feature_columns=feature_columns,
 hidden_units=[256, 32],
 optimizer=tf.train.AdamOptimizer(1e-4),
 n_classes=3,
 dropout=0.1,
 model_dir="../ML/hand_model_5"
)



def predict_with_model(img_data):
    t = np.array(img_data, dtype=np.float16)
    t = t.flatten()
    frame_data = np.array([t])

    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": frame_data},
        y=np.array(['0']),
        num_epochs=1,
        shuffle=False
    )

    return classifier.predict(input_fn = test_input_fn)


def process_raw_predictions(pred_data):
    probs = list(pred_data['predictions'])
    return probs.index(max(probs))



def predict(img_data):
    predictions = predict_with_model(img_data)
    return process_raw_predictions(predictions[0])