# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import random
import os
from scipy.misc import imshow

from model import VAE
from model import BetaVAE
from model import STAE
from data_manager import DataManager
from evaluation import get_ave_recall
from tf_utils import weight_variable, bias_variable

tf.app.flags.DEFINE_integer("epoch_size", 2000, "epoch size")
tf.app.flags.DEFINE_integer("batch_size", 64, "batch size")
tf.app.flags.DEFINE_integer("random_seed", 42, "random seed")
tf.app.flags.DEFINE_float("gamma", 100.0, "gamma param for latent loss")
tf.app.flags.DEFINE_float("capacity_limit", 20.0,
                          "encoding capacity limit param for latent loss")
tf.app.flags.DEFINE_integer("capacity_change_duration", 100000,
                            "encoding capacity change duration")
tf.app.flags.DEFINE_float("learning_rate", 5e-4, "learning rate")
tf.app.flags.DEFINE_string("checkpoint_dir", "checkpoints", "checkpoint directory")
tf.app.flags.DEFINE_string("log_file", "./log", "log file directory")
tf.app.flags.DEFINE_string("model_type", "vae", "model type")
tf.app.flags.DEFINE_string("task_type", "position", "task type")
tf.app.flags.DEFINE_string("dist_type", 'linear', "distribution type")
tf.app.flags.DEFINE_string("expriment_name", "default", "experiment name")
tf.app.flags.DEFINE_boolean("training", True, "training or not")
tf.app.flags.DEFINE_boolean("short_training", False, "training or not")

flags = tf.app.flags.FLAGS

np.set_printoptions(threshold=np.nan)

my_path = "output/" + flags.expriment_name + "/"

ss = 2
xx = 30
yy = 2

def train(sess,
          model,
          manager,
          saver):

  summary_writer = tf.summary.FileWriter(flags.log_file, sess.graph)
  
  n_samples = manager.sample_size

  indices = list(range(n_samples))

  step = 0

  # Training cycle
  for epoch in range(flags.epoch_size):
    print('epoch', epoch)
    # Shuffle image indices
    random.shuffle(indices)
    #indices = manager.get_dependent_indices(n_samples)
    a = []
    rotation_list = []
    for index in indices:
      shape = index // (32 * 32 * 40 * 6)
      scale = (index // (32 * 32 * 40)) % 6
      rotation = (index // (32 * 32)) % 40
      x = (index // 32) % 32
      y = index % 32
      x = 0
      y = 0
      #latents = [0, shape, scale, rotation, x, y]
      latents = [0, ss, 2, rotation, x, y]
      a.append(manager.get_index(latents))
      rotation_list.append(rotation)
    indices = a

    if flags.short_training:
      total_batch = 8000
    else:
      total_batch = n_samples // flags.batch_size

    # Loop over all batches
    avg_cost = 0.0
    avg_acc = 0.0
    for i in range(total_batch):
      # Generate image batch
      batch_indices = indices[flags.batch_size*i : flags.batch_size*(i+1)]
      batch_rotations = rotation_list[flags.batch_size*i : flags.batch_size*(i+1)]
      batch_xs = manager.get_images(batch_indices)

      #j = 11
      #rimg = np.asarray(batch_xs[j]).reshape(64, 64)
      #print(batch_rotations[j])
      #print(rimg.tolist())
      #imshow(rimg*255)
      #exit()

      # Fit training using batch data
      reconstr_loss, accuracy = model.partial_fit(sess, batch_xs, batch_rotations)
      avg_cost += reconstr_loss
      avg_acc += accuracy
      if i % 100 == 0:
        print(i, avg_acc / 100., avg_cost / 100.)
        avg_cost = 0
        avg_acc = 0
      step += 1


def load_checkpoints(sess):
  saver = tf.train.Saver()
  checkpoint = tf.train.get_checkpoint_state(my_path + flags.checkpoint_dir)
  if checkpoint and checkpoint.model_checkpoint_path:
    saver.restore(sess, checkpoint.model_checkpoint_path)
    print("loaded checkpoint: {0}".format(checkpoint.model_checkpoint_path))
  else:
    print("Could not find old checkpoint")
    if not os.path.exists(my_path + flags.checkpoint_dir):
      os.mkdir(my_path + flags.checkpoint_dir)
  return saver

# Model
class RegressionModel(STAE):
  def _create_localization_network2(self, x, reuse=False):
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

      W_fc_loc2 = weight_variable([20, 1])
      # Use identity transformation as starting point
      initial = np.array([0.])
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

  def _create_localization_network(self, x):
    x = tf.layers.dense(x, 20, activation=tf.nn.relu)
    x = tf.layers.dense(x, 40, activation=None)
    return x

  def _create_network(self):
    # tf Graph input
    self.x = tf.placeholder(tf.float32, shape=[None, 4096], name='x')
    self.y = tf.placeholder(tf.int64, shape=[None], name='y')
    self.y_hat = self._create_localization_network(self.x)
    #self.y_hat = tf.reshape(self.y_hat, [-1])

  def _create_loss_optimizer(self):
    # Reconstruction loss
    #self.loss = tf.nn.l2_loss(self.y - self.y_hat)
    self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=self.y_hat)
    self.loss = tf.reduce_mean(self.loss)
    self.reconstr_loss = self.loss
    self.latent_loss = self.loss

    prediction = tf.argmax(self.y_hat, 1)
    equality = tf.equal(prediction, self.y)
    self.accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))

    self.optimizer = tf.train.AdamOptimizer(
      learning_rate=self.learning_rate).minimize(self.loss)

  def partial_fit(self, sess, x, y):
    """Train model based on mini-batch of input data.

    Return loss of mini-batch.
    """
    fetch = [self.optimizer, self.loss, self.accuracy]
    feed = {self.x: x, self.y: y}
    _, loss, accuracy = sess.run(fetch, feed_dict=feed)
    return loss, accuracy


def main(argv):
  seed = flags.random_seed
  random.seed(seed)
  np.random.seed(seed)
  tf.set_random_seed(seed)

  if not os.path.isdir(my_path):
    os.makedirs(my_path)

  if flags.task_type == 'position':
    dims = [32, 32]
  elif flags.task_type == 'shape_position':
    dims = [3, 1, 1, 32, 32]
  elif flags.task_type == 'shape_position_scale':
    dims = [3, 6, 1, 32, 32]
  elif flags.task_type == 'onecolor':
    dims = [3, 6, 40, 32, 32]
  else:
    raise ValueError("Task type is not defined: " + flags.task_type)

  manager = DataManager(flags.dist_type, dims)
  manager.load()

  sess = tf.Session()

  model = RegressionModel()

  sess.run(tf.global_variables_initializer())

  saver = load_checkpoints(sess)

  if flags.training:
    # Train
    train(sess, model, manager, saver)
  else:
    reconstruct_check_images = manager.get_random_images(10)
    # Image reconstruction check
    reconstruct_check(sess, model, reconstruct_check_images)
    # Disentangle check
    disentangle_check(sess, model, manager)
  

if __name__ == '__main__':
  tf.app.run()
