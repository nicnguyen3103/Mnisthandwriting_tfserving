import numpy as np
import tensorflow as tf

#load API from AI platform
class MySimpleScaler(object):

    def preprocess_img(self, data):
        # get data from the front end 
        img_raw = data
        image = tf.image.decode_jpeg(img_raw, channels=1)
        image = tf.image.resize(image, [28, 28])
        image = (255 - image) / 255.0  # normalize to [0,1] range
        image = tf.reshape(image, (1, 28, 28, 1))

        return image