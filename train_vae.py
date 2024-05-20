import tensorflow as tf
import numpy as np


def log_normal_pdf(sample, mean, logvar, raxis=1):
  '''
  This is the logarithm of the probability according to a normal distribution. 
  I.e. log(p(x)) where p is a normal/Gaussian distribution. The naming is a little confusing though.
  '''
  log2pi = tf.math.log(2. * np.pi)
  return tf.reduce_sum(
      -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
      axis=raxis)


def compute_loss(model, x):
  '''
  VAEs train by maximizing the evidence lower bound (ELBO) on the marginal log-likelihood.
  In practice, optimize the single sample Monte Carlo estimate of this expectation:
          log p(x|z) + log p(z) - log q(z|x)  , where z is sampled from q(z|x)
  '''
  mean, logvar = model.encode(x)
  z = model.reparameterize(mean, logvar)
  ## TODO: FLOAT16? OR FLOAT32?
  x_logit = tf.cast(model.decode(z), tf.float32)
  cross_ent = tf.cast(tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x), tf.float32)
  logpx_z = tf.cast(-tf.reduce_sum(cross_ent, axis=[1, 2, 3]), tf.float32)
  logpz = tf.cast(log_normal_pdf(z, 0., 0.), tf.float32)
  logqz_x = tf.cast(log_normal_pdf(z, mean, logvar), tf.float32)
  return -tf.reduce_mean(logpx_z + logpz - logqz_x)


@tf.function
def train_step(model, x, optimizer):
  """Executes one training step and returns the loss.

  This function computes the loss and gradients, and uses the latter to
  update the model's parameters.
  """
  with tf.GradientTape() as tape:
    loss = compute_loss(model, x)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))