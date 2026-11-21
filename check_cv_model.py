import tensorflow as tf
i = tf.lite.Interpreter("models/efficientdet_lite0.tflite"); i.allocate_tensors()
d = i.get_input_details()[0]
print("INPUT:", d["shape"], d["dtype"])
for o in i.get_output_details():
    print("OUTPUT:", o["index"], o["name"], o["shape"])