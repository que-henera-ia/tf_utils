import tensorflow as tf
import tensorflow_docs.vis.embed as embed
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image
import imageio
import glob
import os

def load_image(path_to_img, img_shape=512, img_channels=3, preserve_aspect_ratio=False):
  img = tf.io.read_file(path_to_img)
  img = tf.image.decode_image(img, channels=img_channels)
  img = tf.image.convert_image_dtype(img, tf.float32)

  if preserve_aspect_ratio:
    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = img_shape / long_dim
    new_shape = tf.cast(shape * scale, tf.int32)
  else:
    new_shape = tf.cast([img_shape,img_shape], tf.int32)
  

  img = tf.image.resize(img, new_shape)
  img = img[tf.newaxis, :]
  return img

def load_image_dataset(path_to_dataset, img_shape=512, img_channels=3, preserve_aspect_ratio=False):
  # Check if the path exists
  if not os.path.exists(path_to_dataset):
    print("The specified path does not exist.")
    return
  # Check if the path is a directory
  if not os.path.isdir(path_to_dataset):
    print("The specified path is not a directory.")
    return
  # List all files in the directory
  files = os.listdir(path_to_dataset)
  
  dataset = []
  for file in files:
    img = load_image(os.path.join(path_to_dataset, file), img_shape=img_shape, img_channels=img_channels,preserve_aspect_ratio=preserve_aspect_ratio)
    dataset.append(img[0])
  return tf.convert_to_tensor(dataset)


def extract_video_frames(path_to_video, img_shape=512 ,img_channels=3, preserve_aspect_ratio=False):
  video_reader = imageio.get_reader(path_to_video)
  frames = []
  for frame in video_reader:
    # Convert frame to PIL Image
    pil_frame = PIL.Image.fromarray(frame)
    # Convert to grayscale
    if img_channels==1: 
      pil_frame = pil_frame.convert('L')
    
    pil_frame = tf.convert_to_tensor(np.array(pil_frame))

    if img_channels==1:
      # Extend dimension if grayscale
      pil_frame = pil_frame[..., tf.newaxis]

    if preserve_aspect_ratio:
      shape = tf.cast(tf.shape(pil_frame)[:-1], tf.float32)
      long_dim = max(shape)
      scale = img_shape / long_dim
      new_shape = tf.cast(shape * scale, tf.int32)
    else:
      new_shape = tf.cast([img_shape,img_shape], tf.int32)
    pil_frame = tf.image.resize(pil_frame, new_shape)
    # Append to dataset
    frames.append(pil_frame)

  frames = tf.convert_to_tensor(np.array(frames))
  return frames

def tensor_to_image(tensor):
  tensor = tensor*255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor)>3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  return PIL.Image.fromarray(tensor)

def save_image(image,img_path="data_out/image"):
  fig = plt.figure()
  plt.imshow(image)
  plt.axis('off')
  plt.savefig(img_path + '.png')

def save_image_matrix(images, img_path="data_out/image"):
  fig = plt.figure(figsize=(4, 4))

  for i in range(images.shape[0]):
    plt.subplot(int(np.sqrt(images.shape[0])), int(np.sqrt(images.shape[0])), i + 1)
    plt.imshow(images[i, :, :, :])
    plt.axis('off')

  # tight_layout minimizes the overlap between 2 sub-plots
  plt.savefig(img_path + '.png')


def save_gif(filename, re_images_name='data_out/image*.png'):
  anim_file = filename + '.gif'
  with imageio.get_writer(anim_file, mode='I') as writer:
    filenames = glob.glob(re_images_name)
    filenames = sorted(filenames)
    for filename in filenames:
      image = imageio.imread(filename)
      writer.append_data(image)
    image = imageio.imread(filename)
    writer.append_data(image)
    
  embed.embed_file(anim_file)

def save_mp4(filename, re_images_name='data_out/image*.png'):
  anim_file = filename + '.mp4'
  with imageio.get_writer(anim_file, mode='I') as writer:
    filenames = glob.glob(re_images_name)
    filenames = sorted(filenames)
    for filename in filenames:
      image = imageio.imread(filename)
      writer.append_data(image)
    image = imageio.imread(filename)
    writer.append_data(image)

  embed.embed_file(anim_file)

def split_dataset(data, test_proportion=0.3):
  test_size = int(np.shape(data)[0]*0.1)
  test = data[:test_size]
  train = data[test_size:]
  return train, test

def shuffle_dataset(data):
  return tf.random.shuffle(data)


  '''

def plot_latent_images(model, n, digit_size=28):
  """Plots n x n digit images decoded from the latent space."""

  norm = tfp.distributions.Normal(0, 1)
  grid_x = norm.quantile(np.linspace(0.05, 0.95, n))
  grid_y = norm.quantile(np.linspace(0.05, 0.95, n))
  image_width = digit_size*n
  image_height = image_width
  image = np.zeros((image_height, image_width))

  for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
      z = np.array([[xi, yi]])
      x_decoded = model.sample(z)
      digit = tf.reshape(x_decoded[0], (digit_size, digit_size))
      image[i * digit_size: (i + 1) * digit_size,
            j * digit_size: (j + 1) * digit_size] = digit.numpy()

  plt.figure(figsize=(10, 10))
  plt.imshow(image, cmap='Greys_r')
  plt.axis('Off')
  # plt.show()
'''