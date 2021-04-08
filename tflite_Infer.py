import numpy as np
import tensorflow as tf
import cv2

img = cv2.imread('IM-0001-0001.jpeg')
img = cv2.resize(img,(224,224))
img = np.array(img,dtype="float32")
img = np.reshape(img,(1,224,224,3))

interpreter = tf.lite.Interpreter(model_path="MobiNetmodel.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']

print("*"*50,input_details)
interpreter.set_tensor(input_details[0]['index'], img)

interpreter.invoke()

output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)