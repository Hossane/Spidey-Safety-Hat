import tensorflow as tf
i = tf.lite.Interpreter("models/yamnet.tflite"); i.allocate_tensors()
print(i.get_input_details()[0]["shape"], i.get_input_details()[0]["dtype"])
print(i.get_output_details()[0]["shape"])