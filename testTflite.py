import numpy as np
import tensorflow as tf
import cv2
from PIL import  Image
from skimage import io
with tf.Session() as sess:
    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path="transfer.tflite")
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    content_file = './img/timg.jpg'
    model_path = './transfertransfer.pb'
    output_file = './pbImg.jpg'
    # image_bytes = tf.read_file(content_file)

    img = Image.open(content_file).resize((256, 256))
    input_data = np.reshape(img,[256,256,3])
    input_data = input_data.astype('int32')
    # input_data = np.cast(input_data, np.int32)

    # input_array, decoded_image = sess.run([
    #     tf.reshape(tf.image.decode_jpeg(image_bytes, channels=3), [-1]),
    #     tf.image.decode_jpeg(image_bytes, channels=3)])


    # Test model on random input data.
    input_shape = input_details[0]['shape']
    # input_data = np.array(np.random.random_sample(input_shape), np.int32)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    img = np.reshape(output_data, [input_shape[1], input_shape[0], 3])
    img = np.clip(img, 0, 255).astype(np.uint8)
    io.imsave(output_file, img)

    print(output_data)

    print(input_details)
    print(output_details)