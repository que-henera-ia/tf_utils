import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image
import imageio
import glob

def load_img(path_to_img, img_shape=512):
  max_dim = img_shape
  img = tf.io.read_file(path_to_img)
  img = tf.image.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)

  shape = tf.cast(tf.shape(img)[:-1], tf.float32)
  long_dim = max(shape)
  scale = max_dim / long_dim

  new_shape = tf.cast(shape * scale, tf.int32)

  img = tf.image.resize(img, new_shape)
  img = img[tf.newaxis, :]
  return img

def tensor_to_image(tensor):
  tensor = tensor*255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor)>3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  return PIL.Image.fromarray(tensor)

def save_img(image,img_name):
  fig = plt.figure()
  plt.imshow(image)
  plt.axis('off')
  plt.savefig(img_name + '.png')


def save_gif(filename, re_images_name='img*.png'):
  anim_file = filename + '.gif'
  with imageio.get_writer(anim_file, mode='I') as writer:
    filenames = glob.glob(re_images_name)
    filenames = sorted(filenames)
    for filename in filenames:
      image = imageio.imread(filename)
      writer.append_data(image)
    image = imageio.imread(filename)
    writer.append_data(image)
    

def save_mp4(filename, re_images_name='img*.png'):
  anim_file = filename + '.mp4'
  with imageio.get_writer(anim_file, mode='I') as writer:
    filenames = glob.glob(re_images_name)
    filenames = sorted(filenames)
    for filename in filenames:
      image = imageio.imread(filename)
      writer.append_data(image)
    image = imageio.imread(filename)
    writer.append_data(image)