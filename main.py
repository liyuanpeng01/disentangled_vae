# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import random
import os
from scipy.misc import imsave

from model import VAE
from model import BetaVAE
from model import STAE
from data_manager import DataManager
from evaluation import get_ave_recall

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

  #reconstruct_check_images = manager.get_random_images(10)
  reconstruct_check_images = []
  reconstruct_check_images.append(manager.get_image(0, 2, 0, xx, yy))
  reconstruct_check_images.append(manager.get_image(1, 2, 0, xx, yy))
  reconstruct_check_images.append(manager.get_image(ss, 2, 0, xx, yy))
  reconstruct_check_images.append(manager.get_image(ss, 2, 5, xx, yy))
  reconstruct_check_images.append(manager.get_image(ss, 2, 10, xx, yy))
  reconstruct_check_images.append(manager.get_image(ss, 2, 15, xx, yy))
  reconstruct_check_images.append(manager.get_image(ss, 2, 20, xx, yy))
  reconstruct_check_images.append(manager.get_image(ss, 2, 25, xx, yy))
  reconstruct_check_images.append(manager.get_image(ss, 2, 30, xx, yy))
  reconstruct_check_images.append(manager.get_image(ss, 2, 35, xx, yy))
  reconstruct_check_images.append(manager.get_image(0, 0, 0, xx, yy))
  reconstruct_check_images.append(manager.get_image(0, 1, 0, xx, yy))
  reconstruct_check_images.append(manager.get_image(0, 2, 0, xx, yy))
  reconstruct_check_images.append(manager.get_image(0, 3, 0, xx, yy))
  reconstruct_check_images.append(manager.get_image(0, 4, 0, xx, yy))
  reconstruct_check_images.append(manager.get_image(0, 5, 0, xx, yy))

  #indices = list(range(n_samples))

  step = 0

  # Training cycle
  for epoch in range(flags.epoch_size):
    print('epoch', epoch)
    # Shuffle image indices
    #random.shuffle(indices)
    indices = manager.get_dependent_indices(n_samples)
    a = []
    for index in indices:
      shape = index // (32 * 32 * 40 * 6)
      scale = (index // (32 * 32 * 40)) % 6
      rotation = (index // (32 * 32)) % 40
      x = (index // 32) % 32
      y = index % 32
      #latents = [0, shape, scale, rotation, x, y]
      latents = [0, ss, 2, rotation, x, y]
      a.append(manager.get_index(latents))
    #indices = a

    if flags.short_training:
      total_batch = 2000
    else:
      total_batch = n_samples // flags.batch_size

    # Loop over all batches
    avg_cost = 0.0
    for i in range(total_batch):
      # Generate image batch
      batch_indices = indices[flags.batch_size*i : flags.batch_size*(i+1)]
      batch_xs = manager.get_images(batch_indices)
      # Fit training using batch data
      reconstr_loss, latent_loss, summary_str = model.partial_fit(sess, batch_xs, step)
      avg_cost += reconstr_loss
      if i % 100 == 0:
        print(i, avg_cost / 100.)
        avg_cost = 0

      summary_writer.add_summary(summary_str, step)
      step += 1
    print('reconstruction loss: ', reconstr_loss)

    # Image reconstruction check
    reconstruct_check(sess, model, reconstruct_check_images)

    # Disentangle check
    disentangle_check(sess, model, manager)

    # Transform check
    print("Evaluating transforming.")
    transform_check(sess, model, manager)

    # Save checkpoint
    saver.save(sess, my_path + flags.checkpoint_dir + '/' + 'checkpoint', global_step = step)

    
def reconstruct_check(sess, model, images):
  # Check image reconstruction
  x_reconstruct = model.reconstruct(sess, images)

  if not os.path.exists(my_path + "reconstr_img"):
    os.mkdir(my_path + "reconstr_img")

  for i in range(len(images)):
    org_img = images[i].reshape(64, 64)
    org_img = org_img.astype(np.float32)
    reconstr_img = x_reconstruct[i].reshape(64, 64)
    imsave(my_path + "reconstr_img/org_{0}.png".format(i),      org_img)
    imsave(my_path + "reconstr_img/reconstr_{0}.png".format(i), reconstr_img)


def disentangle_check(sess, model, manager, save_original=False):
  img = manager.get_image(shape=ss, scale=2, orientation=0, x=xx, y=yy)
  if save_original:
    imsave(my_path + "original.png", img.reshape(64, 64).astype(np.float32))
    
  batch_xs = [img]
  z_mean, z_log_sigma_sq = model.transform(sess, batch_xs)
  z_sigma_sq = np.exp(z_log_sigma_sq)[0]

  # Print variance
  zss_str = ""
  for i,zss in enumerate(z_sigma_sq):
    str = "z{0}={1:.4f}".format(i,zss)
    zss_str += str + ", "
  print(zss_str)

  # Save disentangled images
  z_m = z_mean[0]
  n_z = 10

  if not os.path.exists(my_path + "disentangle_img"):
    os.mkdir(my_path + "disentangle_img")

  if flags.model_type == "vae" or flags.model_type == "beta":
    rng = 3.
  elif flags.model_type == "stn":
    rng = 0.5
  else:
    raise ValueError("Model type is not defined: " + flags.model_type)

  for target_z_index in range(n_z):
    for ri in range(n_z):
      value = -rng + ((2 * rng) / 9.0) * ri
      z_mean2 = np.zeros((1, n_z))
      for i in range(n_z):
        if( i == target_z_index ):
          z_mean2[0][i] = value
          if target_z_index == 0:
            z_mean2[0][i] *= 4
        else:
          z_mean2[0][i] = z_m[i]
      reconstr_img = model.generate(sess, z_mean2)
      rimg = reconstr_img[0].reshape(64, 64)
      imsave(my_path + "disentangle_img/check_z{0}_{1}.png".format(target_z_index,ri), rimg)


def transform_check(sess, model, manager):
  a = []
  b = []
  shape = 1
  scale = 2
  orientation = 5

  if flags.task_type == 'position':
    for i in xrange(32):
      batch_xs = []
      for j in xrange(32):
        img = manager.get_image(shape=1, scale=2, orientation=5, x=i, y=j)
        a.append([i, j])
        batch_xs.append(img)
      z_mean, _ = model.transform(sess, batch_xs)
      b.extend(z_mean)

  elif flags.task_type == 'onecolor':
    for shape in xrange(3):
      for scale in xrange(6):
        for orientation in xrange(40):
          batch_xs = []
          for i in xrange(8):
            i *= 4
            for j in xrange(8):
              j *= 4
              img = manager.get_image(
                shape=shape, scale=scale, orientation=orientation, x=i, y=j)
              a.append([scale, orientation, i, j])
              batch_xs.append(img)
          z_mean, _ = model.transform(sess, batch_xs)
          b.extend(z_mean)

  elif flags.task_type == 'shape_position':
    for shape in xrange(3):
      batch_xs = []
      for i in xrange(32):
        for j in xrange(32):
          img = manager.get_image(
            shape=shape, scale=scale, orientation=orientation, x=i, y=j)
          a.append([i, j])
          batch_xs.append(img)
      z_mean, _ = model.transform(sess, batch_xs)
      b.extend(z_mean)

  elif flags.task_type == 'shape_position_scale':
    for shape in xrange(3):
      for scale in xrange(6):
        batch_xs = []
        for i in xrange(32):
          for j in xrange(32):
            img = manager.get_image(
              shape=shape, scale=scale, orientation=orientation, x=i, y=j)
            a.append([scale, i, j])
            batch_xs.append(img)
        z_mean, _ = model.transform(sess, batch_xs)
        b.extend(z_mean)

  else:
    raise ValueError("Task type is not defined: " + flags.task_type)

  for j in range(4):
    with open(my_path + 'align_' + str(j) + '.log', 'w') as f:
      for x, y in zip(a, b):
        f.write(str(x[j]) + '\t' + str(y[j]) + '\n')

  a = np.transpose(a)
  b = np.transpose(b)

  avg_coef, cor_list = get_ave_recall(a, b)
  print("Average correlation coefficient: ", avg_coef)
  print(cor_list)


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

  if flags.model_type == "vae":
    model = VAE(gamma=flags.gamma,
                capacity_limit=flags.capacity_limit,
                capacity_change_duration=flags.capacity_change_duration,
                learning_rate=flags.learning_rate)
  elif flags.model_type == "beta":
    model = BetaVAE(gamma=flags.gamma,
                capacity_limit=flags.capacity_limit,
                capacity_change_duration=flags.capacity_change_duration,
                learning_rate=flags.learning_rate)
  elif flags.model_type == "stn":
    model = STAE(gamma=flags.gamma,
                capacity_limit=flags.capacity_limit,
                capacity_change_duration=flags.capacity_change_duration,
                learning_rate=flags.learning_rate)
  else:
    raise ValueError("Model type is not defined: " + flags.model_type)
  print("Model type is " + flags.model_type)
  
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
