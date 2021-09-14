import tensorflow as tf
import numpy as np

def SSIM_calculate(m_x, m_y, covariance_x_y, variance_x, variance_y):

  #x__ = tf.placeholder(tf.int32, shape=[28,28,1], name='x_placeholder')
  #y__ = tf.placeholder(tf.int32, shape=[28,28,1], name='y_placeholder')

  #input_image = tf.cast(input_image, dtype=tf.float32)
  #output_image = tf.cast(output_image, dtype=tf.float32)

  #input_image = tf.reshape(input_image, shape=[28, 28, 1])
  #output_image = tf.reshape(output_image, shape=[28, 28, 1])

  #x__ = tf.identity(input_image)
  #y__ = tf.identity(output_image)

  c1 = (0.01*255)*(0.01*255)
  c2 = (0.03*255)*(0.03*255)

  numerator = np.multiply(sum(np.multiply(2,m_x,m_y),c1),(sum(np.multiply(2,covariance_x_y),c2)))
  denominator = np.multiply(sum(sum(np.multiply(m_x,m_x),np.multiply(m_y,m_y)),c1),sum(sum(variance_x,variance_y),c2))
  SSIM = numerator/denominator

  return SSIM