import tensorflow as tf
import numpy as np

def SSIM_calculate(input_image, output_image):

  x__ = tf.placeholder(tf.int32, shape=[28,28,1], name='x_placeholder')
  y__ = tf.placeholder(tf.int32, shape=[28,28,1], name='y_placeholder')

  input_image = tf.cast(input_image, dtype=tf.float32)
  output_image = tf.cast(output_image, dtype=tf.float32)

  input_image = tf.reshape(input_image, shape=[28, 28, 1])
  output_image = tf.reshape(output_image, shape=[28, 28, 1])

  x__ = tf.identity(input_image)
  y__ = tf.identity(output_image)

  mean_x, variance_x = tf.nn.moments(x__, [0])
  mean_y, variance_y = tf.nn.moments(y__, [0])

  c1 = (0.01*255)*(0.01*255)
  c2 = (0.03*255)*(0.03*255)

  x_y_covariance, x_y_optimiser = tf.contrib.metrics.streaming_covariance(x__, y__)
  x_x_covariance, x_x_optimiser = tf.contrib.metrics.streaming_covariance(x__, x__)
  y_y_covariance, y_y_optimiser = tf.contrib.metrics.streaming_covariance(y__, y__)

  sess_inner = tf.Session()

  sess_inner.run(tf.global_variables_initializer())
  sess_inner.run(tf.local_variables_initializer())
  m_x, _ = sess_inner.run([mean_x, variance_x])
  m_y, _ = sess_inner.run([mean_y, variance_y])

  sess_inner.run([x_y_optimiser])
  covariance_x_y = sess_inner.run([x_y_covariance])
  sess_inner.run([x_x_optimiser])
  variance_x = sess_inner.run([x_x_covariance])
  sess_inner.run([y_y_optimiser])
  variance_y = sess_inner.run([y_y_covariance])

  numerator = np.multiply(sum(np.multiply(2,m_x,m_y),c1),(sum(np.multiply(2,covariance_x_y),c2)))
  denominator = np.multiply(sum(sum(np.multiply(m_x,m_x),np.multiply(m_y,m_y)),c1),sum(sum(variance_x,variance_y),c2))
  SSIM = numerator/denominator

  return SSIM