import tensorflow as tf
from tensorflow.python.keras.models import load_model

model = load_model('./models/xy_wh_pp.h5')

# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open('xy_wh_pp.tflite', 'wb') as f:
  f.write(tflite_model)

