# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import math
import tensorflow as tf
from spatial_transformer import transformer
from tf_utils import weight_variable, bias_variable


def fc_initializer(input_channels, dtype=tf.float32):
  def _initializer(shape, dtype=dtype, partition_info=None):
    d = 1.0 / np.sqrt(input_channels)
    return tf.random_uniform(shape, minval=-d, maxval=d)
  return _initializer


def conv_initializer(kernel_width, kernel_height, input_channels, dtype=tf.float32):
  def _initializer(shape, dtype=dtype, partition_info=None):
    d = 1.0 / np.sqrt(input_channels * kernel_width * kernel_height)
    return tf.random_uniform(shape, minval=-d, maxval=d)
  return _initializer


class VAE(object):
  """ Beta Variational Auto Encoder. """
  
  def __init__(self,
               gamma=100.0,
               capacity_limit=25.0,
               capacity_change_duration=100000,
               learning_rate=5e-4):
    self.gamma = gamma
    self.capacity_limit = capacity_limit
    self.capacity_change_duration = capacity_change_duration
    self.learning_rate = learning_rate
    
    # Create autoencoder network
    self._create_network()
    
    # Define loss function and corresponding optimizer
    self._create_loss_optimizer()

  def _conv2d_weight_variable(self, weight_shape, name, deconv=False):
    name_w = "W_{0}".format(name)
    name_b = "b_{0}".format(name)
    
    w = weight_shape[0]
    h = weight_shape[1]
    if deconv:
      input_channels  = weight_shape[3]
      output_channels = weight_shape[2]
    else:
      input_channels  = weight_shape[2]
      output_channels = weight_shape[3]
    d = 1.0 / np.sqrt(input_channels * w * h)
    bias_shape = [output_channels]

    weight = tf.get_variable(name_w, weight_shape,
                             initializer=conv_initializer(w, h, input_channels))
    bias   = tf.get_variable(name_b, bias_shape,
                             initializer=conv_initializer(w, h, input_channels))
    return weight, bias


  def _fc_weight_variable(self, weight_shape, name):
    name_w = "W_{0}".format(name)
    name_b = "b_{0}".format(name)
    
    input_channels  = weight_shape[0]
    output_channels = weight_shape[1]
    d = 1.0 / np.sqrt(input_channels)
    bias_shape = [output_channels]

    weight = tf.get_variable(name_w, weight_shape, initializer=fc_initializer(input_channels))
    bias   = tf.get_variable(name_b, bias_shape,   initializer=fc_initializer(input_channels))
    return weight, bias
  
  
  def _get_deconv2d_output_size(self, input_height, input_width, filter_height,
                                filter_width, row_stride, col_stride, padding_type):
    if padding_type == 'VALID':
      out_height = (input_height - 1) * row_stride + filter_height
      out_width  = (input_width  - 1) * col_stride + filter_width
    elif padding_type == 'SAME':
      out_height = input_height * row_stride
      out_width  = input_width * col_stride
    return out_height, out_width
  
  
  def _conv2d(self, x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1],
                        padding='SAME')
  
  
  def _deconv2d(self, x, W, input_width, input_height, stride):
    filter_height = W.get_shape()[0].value
    filter_width  = W.get_shape()[1].value
    out_channel   = W.get_shape()[2].value
    
    out_height, out_width = self._get_deconv2d_output_size(input_height,
                                                           input_width,
                                                           filter_height,
                                                           filter_width,
                                                           stride,
                                                           stride,
                                                           'SAME')
    batch_size = tf.shape(x)[0]
    output_shape = tf.stack([batch_size, out_height, out_width, out_channel])
    return tf.nn.conv2d_transpose(x, W, output_shape,
                                  strides=[1, stride, stride, 1],
                                  padding='SAME')

  def _sample_z(self, z_mean, z_log_sigma_sq):
    eps_shape = tf.shape(z_mean)
    eps = tf.random_normal( eps_shape, 0, 1, dtype=tf.float32 )
    # z = mu + sigma * epsilon
    z = tf.add(z_mean,
               tf.multiply(tf.sqrt(tf.exp(z_log_sigma_sq)), eps))
    return z
    
  
  def _create_recognition_network(self, x, reuse=False):
    with tf.variable_scope("rec", reuse=reuse) as scope:
      # [filter_height, filter_width, in_channels, out_channels]
      W_conv1, b_conv1 = self._conv2d_weight_variable([4, 4, 1,  32], "conv1")
      W_conv2, b_conv2 = self._conv2d_weight_variable([4, 4, 32, 32], "conv2")
      W_conv3, b_conv3 = self._conv2d_weight_variable([4, 4, 32, 32], "conv3")
      W_conv4, b_conv4 = self._conv2d_weight_variable([4, 4, 32, 32], "conv4")
      W_fc1, b_fc1     = self._fc_weight_variable([4*4*32, 256], "fc1")
      W_fc2, b_fc2     = self._fc_weight_variable([256, 256], "fc2")
      W_fc3, b_fc3     = self._fc_weight_variable([256, 10],  "fc3")
      W_fc4, b_fc4     = self._fc_weight_variable([256, 10],  "fc4")

      x_reshaped = tf.reshape(x, [-1, 64, 64, 1])
      h_conv1 = tf.nn.relu(self._conv2d(x_reshaped, W_conv1, 2) + b_conv1) # (32, 32)
      h_conv2 = tf.nn.relu(self._conv2d(h_conv1,    W_conv2, 2) + b_conv2) # (16, 16)
      h_conv3 = tf.nn.relu(self._conv2d(h_conv2,    W_conv3, 2) + b_conv3) # (8, 8)
      h_conv4 = tf.nn.relu(self._conv2d(h_conv3,    W_conv4, 2) + b_conv4) # (4, 4)
      h_conv4_flat = tf.reshape(h_conv4, [-1, 4*4*32])
      h_fc1 = tf.nn.relu(tf.matmul(h_conv4_flat, W_fc1) + b_fc1)
      h_fc2 = tf.nn.relu(tf.matmul(h_fc1,        W_fc2) + b_fc2)
      z_mean         = tf.matmul(h_fc2, W_fc3) + b_fc3
      z_log_sigma_sq = tf.matmul(h_fc2, W_fc4) + b_fc4
      return (z_mean, z_log_sigma_sq)

  
  def _create_generator_network(self, z, reuse=False):
    with tf.variable_scope("gen", reuse=reuse) as scope:
      W_fc1, b_fc1 = self._fc_weight_variable([10,  256],    "fc1")
      W_fc2, b_fc2 = self._fc_weight_variable([256, 4*4*32], "fc2")

      # [filter_height, filter_width, output_channels, in_channels]
      W_deconv1, b_deconv1 = self._conv2d_weight_variable([4, 4, 32, 32], "deconv1", deconv=True)
      W_deconv2, b_deconv2 = self._conv2d_weight_variable([4, 4, 32, 32], "deconv2", deconv=True)
      W_deconv3, b_deconv3 = self._conv2d_weight_variable([4, 4, 32, 32], "deconv3", deconv=True)
      W_deconv4, b_deconv4 = self._conv2d_weight_variable([4, 4,  1, 32], "deconv4", deconv=True)

      h_fc1 = tf.nn.relu(tf.matmul(z,     W_fc1) + b_fc1)
      h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
      h_fc2_reshaped = tf.reshape(h_fc2, [-1, 4, 4, 32])
      h_deconv1   = tf.nn.relu(self._deconv2d(h_fc2_reshaped, W_deconv1,  4,  4, 2) + b_deconv1)
      h_deconv2   = tf.nn.relu(self._deconv2d(h_deconv1,      W_deconv2,  8,  8, 2) + b_deconv2)
      h_deconv3   = tf.nn.relu(self._deconv2d(h_deconv2,      W_deconv3, 16, 16, 2) + b_deconv3)
      h_deconv4   =            self._deconv2d(h_deconv3,      W_deconv4, 32, 32, 2) + b_deconv4
      
      x_out_logit = tf.reshape(h_deconv4, [-1, 64*64*1])
      return x_out_logit

    
  def _create_network(self):
    # tf Graph input
    self.x = tf.placeholder(tf.float32, shape=[None, 4096])
    
    with tf.variable_scope("vae"):
      self.z_mean, self.z_log_sigma_sq = self._create_recognition_network(self.x)

      # Draw one sample z from Gaussian distribution
      # z = mu + sigma * epsilon
      self.z = self._sample_z(self.z_mean, self.z_log_sigma_sq)
      self.x_out_logit = self._create_generator_network(self.z)
      self.x_out = tf.nn.sigmoid(self.x_out_logit)
      
      
  def _create_loss_optimizer(self):
    # Reconstruction loss
    reconstr_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.x,
                                                            logits=self.x_out_logit)
    reconstr_loss = tf.reduce_sum(reconstr_loss, 1)
    self.reconstr_loss = tf.reduce_mean(reconstr_loss)

    # Latent loss
    latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq
                                       - tf.square(self.z_mean)
                                       - tf.exp(self.z_log_sigma_sq), 1)
    self.latent_loss = tf.reduce_mean(latent_loss)
    
    # Encoding capcity
    self.capacity = tf.placeholder(tf.float32, shape=[])
    
    # Loss with encoding capacity term
    self.loss = self.reconstr_loss + self.gamma * tf.abs(self.latent_loss - self.capacity)

    reconstr_loss_summary_op = tf.summary.scalar('reconstr_loss', self.reconstr_loss)
    latent_loss_summary_op   = tf.summary.scalar('latent_loss',   self.latent_loss)
    self.summary_op = tf.summary.merge([reconstr_loss_summary_op, latent_loss_summary_op])

    self.optimizer = tf.train.AdamOptimizer(
      learning_rate=self.learning_rate).minimize(self.loss)


  def _calc_encoding_capacity(self, step):
    if step > self.capacity_change_duration:
      c = self.capacity_limit
    else:
      c = self.capacity_limit * (step / self.capacity_change_duration)
    return c

    
  def partial_fit(self, sess, xs, step):
    """Train model based on mini-batch of input data.
    
    Return loss of mini-batch.
    """
    c = self._calc_encoding_capacity(step)
    _, reconstr_loss, latent_loss, summary_str = sess.run((self.optimizer,
                                                           self.reconstr_loss,
                                                           self.latent_loss,
                                                           self.summary_op),
                                                          feed_dict={
                                                            self.x : xs,
                                                            self.capacity : c
                                                          })
    return reconstr_loss, latent_loss, summary_str


  def reconstruct(self, sess, xs):
    """ Reconstruct given data. """
    # Original VAE output
    return sess.run(self.x_out, 
                    feed_dict={self.x: xs})

  
  def transform(self, sess, xs):
    """Transform data by mapping it into the latent space."""
    return sess.run( [self.z_mean, self.z_log_sigma_sq],
                     feed_dict={self.x: xs} )
  

  def generate(self, sess, zs):
    """ Generate data by sampling from latent space. """
    return sess.run( self.x_out,
                     feed_dict={self.z: zs} )


class BetaVAE(VAE):
  """ Beta Variational Auto Encoder. """

  def _create_loss_optimizer(self):
    # Reconstruction loss
    reconstr_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.x,
                                                            logits=self.x_out_logit)
    reconstr_loss = tf.reduce_sum(reconstr_loss, 1)
    self.reconstr_loss = tf.reduce_mean(reconstr_loss)

    # Latent loss
    latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq
                                       - tf.square(self.z_mean)
                                       - tf.exp(self.z_log_sigma_sq), 1)
    self.latent_loss = tf.reduce_mean(latent_loss)

    # Encoding capcity
    self.capacity = tf.placeholder(tf.float32, shape=[])

    # Loss with encoding capacity term
    self.loss = self.reconstr_loss + self.gamma * self.latent_loss

    reconstr_loss_summary_op = tf.summary.scalar('reconstr_loss',
                                                 self.reconstr_loss)
    latent_loss_summary_op = tf.summary.scalar('latent_loss', self.latent_loss)
    self.summary_op = tf.summary.merge(
      [reconstr_loss_summary_op, latent_loss_summary_op])

    self.optimizer = tf.train.AdamOptimizer(
      learning_rate=self.learning_rate).minimize(self.loss)


class STAE(VAE):
  """ Beta Variational Auto Encoder. """


  def _create_localization_network(self, x, reuse=False):
    with tf.variable_scope("localization", reuse=reuse) as scope:
      # %% Since x is currently [batch, height*width], we need to reshape to a
      # 4-D tensor to use it in a convolutional graph.  If one component of
      # `shape` is the special value -1, the size of that dimension is
      # computed so that the total size remains constant.  Since we haven't
      # defined the batch dimension's shape yet, we use -1 to denote this
      # dimension should not change size.
      # x_tensor = tf.reshape(x, [-1, 64, 64, 1])

      # %% We'll setup the two-layer localisation network to figure out the
      # %% parameters for an affine transformation of the input
      # %% Create variables for fully connected layer
      W_fc_loc1 = weight_variable([64 * 64, 20])
      b_fc_loc1 = bias_variable([20])

      W_fc_loc2 = weight_variable([20, 4])
      # Use identity transformation as starting point
      initial = np.array([0., 1., 0., 0.])
      initial = initial.astype('float32')
      initial = initial.flatten()
      b_fc_loc2 = tf.Variable(initial_value=initial, name='b_fc_loc2')

      # %% Define the two layer localisation network
      h_fc_loc1 = tf.nn.tanh(tf.matmul(x, W_fc_loc1) + b_fc_loc1)
      # %% We can add dropout for regularizing and to reduce overfitting like so:
      # keep_prob = tf.placeholder(tf.float32)
      h_fc_loc1_drop = h_fc_loc1
      # h_fc_loc1_drop = tf.nn.dropout(h_fc_loc1, keep_prob)
      # %% Second layer
      h_fc_loc2 = tf.nn.tanh(tf.matmul(h_fc_loc1_drop, W_fc_loc2) + b_fc_loc2)
      # theta = tf.split(h_fc_loc2, 4, axis=1)
      theta = h_fc_loc2
    return theta

  def _get_rotation_matrix(self, phi):
    cos = tf.cos(phi)
    sin = tf.sin(phi)

    zero = 0. * phi
    one = zero + 1.

    matrix = tf.stack([cos, -sin, zero, sin, cos, zero, zero, zero, one],
                      axis=1)
    matrix = tf.reshape(matrix, [-1, 3, 3])
    return matrix

  def _get_scaling_matrix(self, s):
    zero = 0. * s
    one = zero + 1.

    matrix = tf.stack([s, zero, zero, zero, s, zero, zero, zero, one], axis=1)
    matrix = tf.reshape(matrix, [-1, 3, 3])
    return matrix

  def _get_translation_matrix(self, tx, ty):
    zero = 0. * tx
    one = zero + 1.
    matrix = tf.stack([one, zero, tx, zero, one, ty, zero, zero, one], axis=1)
    matrix = tf.reshape(matrix, [-1, 3, 3])
    return matrix

  def _get_matrix(self, theta, inverse=False):
    phi, s, tx, ty = tf.split(theta, 4, axis=1)

    if inverse:
      phi = -phi
      s = 1. / s
      tx = -tx
      ty = -ty

    #R = self._get_rotation_matrix(phi)
    S = self._get_scaling_matrix(s)
    T = self._get_translation_matrix(tx, ty)
    #order = [T, S, R]
    order = [T, S]

    if inverse:
      order.reverse()

    m = order[0]
    for x in order[1:]:
      m = tf.matmul(m, x)
    #m = tf.matmul(order[1], order[2])
    #m = tf.matmul(order[0], m)
    m = tf.split(m, [2, 1], axis=1)
    m = tf.reshape(m[0], [-1, 6])
    return m

  def ff_network(self, x, n_layers, hidden_size, out_size):
    for i in range(n_layers):
      if i == n_layers - 1:
        activation = None
      else:
        activation = tf.nn.relu
      x = tf.layers.dense(x, out_size, activation=activation)
    return x

  def _create_network(self):
    # tf Graph input
    self.x = tf.placeholder(tf.float32, shape=[None, 4096], name='x')

    # localization
    theta = self._create_localization_network(self.x)

    with tf.variable_scope("input_transform_matrix"):
      A = self._get_matrix(theta)

    x_tensor = tf.reshape(self.x, [-1, 64, 64, 1])
    csize = 32
    hsize = 32
    out_size = (csize, csize)
    c = transformer(x_tensor, A, out_size)

    with tf.variable_scope("encoder"):
      c_flat = tf.reshape(c, [-1, csize * csize])
      h = self.ff_network(c_flat, 1, 16, 6)

    self.z = tf.concat([theta, h], axis=1, name='z')

    theta2, h2 = tf.split(self.z, [4, 6], axis=1)
    # h2 = tf.nn.dropout(h2, 0.2)

    with tf.variable_scope("decoder"):
      c_hat_flat = self.ff_network(h2, 1, 16, hsize * hsize)
      c_hat = tf.reshape(c_hat_flat, [-1, hsize, hsize, 1])
      #c_hat = tf.nn.sigmoid(c_hat)
      pad_size = 64 - hsize
      paddings = [[0, 0], [0, pad_size], [0, pad_size], [0, 0]]
      c_hat = tf.pad(c_hat, paddings)

    with tf.variable_scope("output_transform_matrix"):
      A_inv = self._get_matrix(theta2, inverse=True)
    x_hat = transformer(c_hat, A_inv, (64, 64))

    self.x_out = tf.reshape(x_hat, [-1, 64 * 64], name="x_out")

    self.z_mean = self.z
    self.z_log_sigma_sq = self.z

  def _create_loss_optimizer(self):
    # Reconstruction loss
    #reconstr_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.x,
    #                                                        logits=self.x_out_logit)
    # reconstr_loss = tf.reduce_sum(reconstr_loss, 1)
    #reconstr_loss = tf.losses.sigmoid_cross_entropy(self.x, self.x_out)
    reconstr_loss = tf.nn.l2_loss(self.x - self.x_out)

    self.reconstr_loss = tf.reduce_mean(reconstr_loss)

    # Latent loss
    # latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq
    #                                   - tf.square(self.z_mean)
    #                                   - tf.exp(self.z_log_sigma_sq), 1)
    # self.latent_loss = tf.reduce_mean(latent_loss)
    self.latent_loss = tf.constant(0)

    # Encoding capcity
    self.capacity = tf.placeholder(tf.float32, shape=[])

    # Loss with encoding capacity term
    # self.loss = self.reconstr_loss + self.gamma * tf.abs(self.latent_loss - self.capacity)
    self.loss = self.reconstr_loss

    reconstr_loss_summary_op = tf.summary.scalar('reconstr_loss',
                                                 self.reconstr_loss)
    latent_loss_summary_op = tf.summary.scalar('latent_loss', self.latent_loss)
    self.summary_op = tf.summary.merge(
      [reconstr_loss_summary_op, latent_loss_summary_op])

    self.optimizer = tf.train.AdamOptimizer(
      learning_rate=self.learning_rate).minimize(self.loss)
