import tensorflow as tf
import numpy as np


def clip_0_1(image):
  return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

def normalize_images(images):
  # Normalize the image by 255
  return tf.convert_to_tensor(np.array(images) / 255.0, dtype="float32")