import tensorflow as tf
import numpy as np
import sys
import glob
import os
import random

# Sources List:
# Read input from file
# 1) https://agray3.github.io/2016/11/29/Demystifying-Data-Input-to-TensorFlow-for-Deep-Learning.html
# 2) https://github.com/tensorflow/models/blob/master/research/inception/inception/data/build_image_data.py
# 3) https://github.com/affinelayer/pix2pix-tensorflow
# Neural Network Structure
# 4) https://software.intel.com/en-us/articles/an-example-of-a-convolutional-neural-network-for-image-super-resolution-tutorial
# CNN Implementation
# 5) http://adventuresinmachinelearning.com/convolutional-neural-networks-tutorial-tensorflow/
# 6) http://adventuresinmachinelearning.com/python-tensorflow-tutorial/

# Input images should be 2 (n*n) sized images next to each other. Total image size should be 2n*n
# [image_to_be_filtered][desired_output_image]

#Python optimisation variables
learning_rate = 0.0001
epochs = 100
batch_size = 586

#Used for image input
class ImageCoder(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Create a single Session to run all image coding calls.
    self._sess = tf.Session()

    # Initializes function that converts PNG to JPEG data.
    self._png_data = tf.placeholder(dtype=tf.string)
    image = tf.image.decode_png(self._png_data, channels=3)
    self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)

    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  def png_to_jpeg(self, image_data):
    return self._sess.run(self._png_to_jpeg,
                          feed_dict={self._png_data: image_data})

  def decode_jpeg(self, image_data):
    image = self._sess.run(self._decode_jpeg,
                           feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image

def _find_image_files(data_dir):
  """Build a list of all images files and labels in the data set.
  Args:
    data_dir: string, path to the root directory of images.
    Assume images are all .png
  Returns:
    filenames: list of strings; each string is a path to an image file.
  """

  #construct the list of filenames
  filenames = glob.glob(os.path.join(data_dir, "*.png"))

  # Shuffle the ordering of all image files in order to guarantee
  # random ordering of the images with respect to label in the
  # saved TFRecord files. Make the randomization repeatable.
  shuffled_index = list(range(len(filenames)))
  random.seed(12345)
  random.shuffle(shuffled_index)

  filenames = [filenames[i] for i in shuffled_index]
  return filenames

#Modified function that takes the complete image and separates it into unfiltered and expected datasets
def _process_image(filename):
  """Process a single image file.
  Args:
    filename: string, path to an image file e.g., '/path/to/example.JPG'.
  Returns:
    input_image_return: image to be filtered
    output_image_return: desired image output
  """
  # Read the image file.
  with tf.gfile.FastGFile(filename, 'rb') as f:
    image_data = f.read()

  coder = ImageCoder()
  # Convert any PNG files to JPEG files
  image_data = coder.png_to_jpeg(image_data)

  # Decode the RGB JPEG.
  image = coder.decode_jpeg(image_data)
  #take from 3 channels to 1 (greyscale) (because processing time)
  image = tf.image.rgb_to_grayscale(image)

  # Convert to Tensor
  image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  # Set shape into a form used in next step
  image.set_shape([None, None, 1])

  # Split complete image into 2 images. For an explanation see source 3)
  width = tf.shape(image)[1]  # [height, width, channels]
  input_image_return = ((image[:, :width // 2, :])*2-1)
  output_image_return = ((image[:, width // 2:, :])*2-1)

  return input_image_return, output_image_return

#To do: make this as passing an argument, but that is 0% a priority
filenames = _find_image_files("C:\\Users\\HWRacing\\git\\AdvancedReading\\training_data")
#Unfiltered image
image_input=[]
#Expected output image
image_output=[]

#For each image, get the image and process
#Save list of input images and output images
counter = 0
for i in filenames:
    counter = counter + 1
    print("i=",counter)
    input, output = _process_image(i)
    image_input.append(input)
    image_output.append(output)

image_input_batch=[]
image_output_batch=[]

#Used for generating the test output which is saved to a file
image_input_test=[]
image_output_test=[]

origin_dir = "C:\\Users\\HWRacing\\git\\AdvancedReading\\input_data\\"
image_input_test.append(image_input[0])
image_output_test.append(image_output[0])
png = tf.image.encode_png(tf.cast((tf.reshape(image_input[0], [28, 28, 1])+1.)*127.5,tf.uint8))
sess = tf.Session()
_output_png = sess.run(png)
input_filename = origin_dir+"input_image.png"
open(input_filename, 'wb').write(_output_png)

#Function used to create the convolutional layers of the neural network, and saves repeating code
def create_new_conv_layer(input_data, num_input_channels, num_filters, filter_shape, name):
    #setup the filter input shape for tf.nn.conv_2d
    conv_filt_shape = [filter_shape[0], filter_shape[1], num_input_channels, num_filters]

    #initialise weights and bias for the filter
    weights = tf.Variable(tf.truncated_normal(conv_filt_shape, stddev=0.03), name=name+'_W')
    bias = tf.Variable(tf.truncated_normal([num_filters]), name=name+'_b')

    #setup the convolutional layer operation
    out_layer = tf.nn.conv2d(input_data, weights, [1,1,1,1], padding='SAME')

    #add the bias
    out_layer += bias

    #apply a ReLU non-linear activation
    out_layer = tf.nn.relu(out_layer)

    return out_layer

#Input and output image placeholders
#None = unknown size of list yet, determined by network
#28 x 28 pixels
#1 = 1 colour channel (greyscale) as opposed to 3 (RGB)
x = tf.placeholder(tf.float32, [None, 28, 28, 1])
y = tf.placeholder(tf.float32, [None, 28, 28, 1])

#create some convolutional layers
patch_extraction = create_new_conv_layer(x, 1, 64, [9, 9], name='patch_extraction')
non_linear_mapping = create_new_conv_layer(patch_extraction, 64, 32, [1, 1], name='non_linear_mapping')
reconstruction = create_new_conv_layer(non_linear_mapping, 32, 1, [5, 5], name='reconstruction')

#Reconstruct the output image into a 28x28x1 array
y_ = tf.reshape(reconstruction, [-1, 28, 28, 1])

#Cost function - mean squared error
cross_entropy = tf.reduce_mean(tf.square(y_ - y))

#define an accurate assessment operation
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

#Algorithm for computing mean squared error
#Comes from https://github.com/tensorflow/tensorflow/issues/1666
def log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator

def psnr(accuracy, INPUT_Y, INPUT_Y_):
    rmse = tf.sqrt(accuracy)
    final_accuracy = 20 * log10(255.0 / rmse)
    return final_accuracy

accuracy_old = tf.reduce_mean(tf.square(tf.cast(correct_prediction, tf.float32)))
accuracy = psnr(accuracy_old, y, y_)

#Full credit goes to https://stackoverflow.com/questions/39051451/ssim-ms-ssim-for-tensorflow - I tried but could not implement it myself!
def _tf_fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x_data, y_data = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]

    x_data = np.expand_dims(x_data, axis=-1)
    x_data = np.expand_dims(x_data, axis=-1)

    y_data = np.expand_dims(y_data, axis=-1)
    y_data = np.expand_dims(y_data, axis=-1)

    x = tf.constant(x_data, dtype=tf.float32)
    y = tf.constant(y_data, dtype=tf.float32)

    g = tf.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g / tf.reduce_sum(g)

def tf_ssim(img1, img2, cs_map=False, mean_metric=True, size=11, sigma=1.5):
    window = _tf_fspecial_gauss(size, sigma) # window shape [size, size]
    K1 = 0.01
    K2 = 0.03
    L = 1  # depth of image (255 in case the image has a differnt scale)
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    mu1 = tf.nn.conv2d(img1, window, strides=[1,1,1,1], padding='VALID')
    mu2 = tf.nn.conv2d(img2, window, strides=[1,1,1,1],padding='VALID')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = tf.nn.conv2d(img1*img1, window, strides=[1,1,1,1],padding='VALID') - mu1_sq
    sigma2_sq = tf.nn.conv2d(img2*img2, window, strides=[1,1,1,1],padding='VALID') - mu2_sq
    sigma12 = tf.nn.conv2d(img1*img2, window, strides=[1,1,1,1],padding='VALID') - mu1_mu2
    if cs_map:
        value = (((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2)),
                (2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2))
    else:
        value = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2))

    if mean_metric:
        value = tf.reduce_mean(value)
    return value

#Evaluation function definition
loss = tf_ssim(y, y_)
#Apply an optimizer to mean squared error
optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

#set up the initialisation operator
init_op = tf.global_variables_initializer()

#saver for model
saver = tf.train.Saver()

#set up recording variables
#add a summary to store the cost and evaluation functions
tf.summary.scalar('MSE', cross_entropy)
tf.summary.scalar('SSIM', loss)
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('C:\\Users\\HWRacing\\TensorTest\\Complete_Run')

sess = tf.InteractiveSession()

#set up the initialisation operator
sess.run(tf.global_variables_initializer())

png = tf.image.encode_png(tf.cast((tf.reshape(y_[0], [28, 28, 1])+1.)*127.5,tf.uint8))

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess,coord=coord)

total_batch = batch_size
#Actually perform the neural network training
for epoch in range(epochs):
    avg_cost = 0
    for i in range(total_batch):
        batch_xs, batch_ys = sess.run([image_input, image_output])
        #Optimizer has no usable output but still needs run, save the cross entropy and the loss function
        _, c, acc = sess.run([optimiser, cross_entropy, loss], feed_dict={x: batch_xs, y: batch_ys})
        #for item in acc:
        avg_cost += acc/total_batch
        print("Average cost = ",avg_cost," at ",i," with epoch ",epoch)
    #Once the network has been trained with each batch, test it to find the output
    test_acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys})
    #Print values to the console
    print("Epoch:", (epoch+1), "cost =", "{:.3f}".format(avg_cost), " test accuracy: {:.3f}".format(test_acc))
    #Save the values using TensorBoard
    summary = sess.run(merged, feed_dict={x:batch_xs, y: batch_ys})
    writer.add_summary(summary, epoch)
    #Test the network using the input data and save the output to a file
    output_dir = "C:\\Users\\HWRacing\\git\\AdvancedReading\\output_data\\"
    batch_xs_test, batch_ys_test = sess.run([image_input_test, image_output_test])
    _, acc, cross, _png_data = sess.run([optimiser, accuracy, cross_entropy, png], feed_dict={x: batch_xs_test, y: batch_ys_test})
    output_filename = output_dir + "output" + str(epoch) + ".png"
    open(output_filename, 'wb').write(_png_data)

print("Training complete!")
print("Final SSIM")
#saver.save(sess, 'C:\\Users\\HWRacing\\AppData\\Local\\Programs\\Python\\Python36\\Scripts\\complete_run')
#print("Finished saving")
writer.add_graph(sess.graph)
print(sess.run(loss, feed_dict={x: batch_xs, y: batch_ys}))