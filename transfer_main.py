# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import random
import os

from model import VAE
from model import BetaVAE
from model import STAE
from data_manager import DataManager
from real_data_manager import RealImageGenerator

tf.app.flags.DEFINE_float("gamma", 100.0, "gamma param for latent loss")
tf.app.flags.DEFINE_float("capacity_limit", 20.0,
                          "encoding capacity limit param for latent loss")
tf.app.flags.DEFINE_integer("capacity_change_duration", 100000,
                            "encoding capacity change duration")
tf.app.flags.DEFINE_string("task_type", "position", "task type")
tf.app.flags.DEFINE_boolean("training", True, "training or not")
tf.app.flags.DEFINE_boolean("short_training", False, "training or not")
tf.app.flags.DEFINE_boolean("rep_regularize", False, "regularize representation")
tf.app.flags.DEFINE_boolean("rep_regularize_l1", False, "regularize representation")
tf.app.flags.DEFINE_boolean("sigmoid_output", False, "use sigmoid output")
tf.app.flags.DEFINE_integer("compact_hidden", 6, "number of compact hidden units")
tf.app.flags.DEFINE_boolean("real_data", False, "use real data")

tf.app.flags.DEFINE_integer("epoch_size", 1, "epoch size")
tf.app.flags.DEFINE_integer("batch_size", 64, "batch size")
tf.app.flags.DEFINE_integer("random_seed", 42, "random seed")
tf.app.flags.DEFINE_float("learning_rate", 5e-4, "learning rate")
tf.app.flags.DEFINE_string("checkpoint_dir", "checkpoints", "checkpoint directory")
tf.app.flags.DEFINE_string("log_file", "./log", "log file directory")
tf.app.flags.DEFINE_string("model_type", "vae", "model type")
tf.app.flags.DEFINE_string("dist_type", 'linear', "distribution type")
tf.app.flags.DEFINE_string("expriment_name", "default", "experiment name")
tf.app.flags.DEFINE_float("alpha", 0.1, "alpha")
tf.app.flags.DEFINE_integer("task_id", 0, "task id")
tf.app.flags.DEFINE_string("source_model", "", "source model directory")
tf.app.flags.DEFINE_string("target_model", "", "target model directory")

flags = tf.app.flags.FLAGS

my_path = "output/" + flags.expriment_name + "/"

def load_checkpoints(sess, model_path):
  saver = tf.train.Saver()
  checkpoint = tf.train.get_checkpoint_state(model_path)
  if checkpoint and checkpoint.model_checkpoint_path:
    saver.restore(sess, checkpoint.model_checkpoint_path)
    print("loaded checkpoint: {0}".format(checkpoint.model_checkpoint_path))
  else:
    print("Could not find old checkpoint")
  return saver

def extract_features(manager, model_type, model_path, dims, real_data):
  manager.load()

  sess = tf.Session()

  if model_type == "vae":
    model = VAE(gamma=flags.gamma,
                capacity_limit=flags.capacity_limit,
                capacity_change_duration=flags.capacity_change_duration,
                learning_rate=flags.learning_rate, flags=flags, real_data=real_data)
  elif model_type == "beta":
    model = BetaVAE(gamma=flags.gamma,
                    capacity_limit=flags.capacity_limit,
                    capacity_change_duration=flags.capacity_change_duration,
                    learning_rate=flags.learning_rate, flags=flags, real_data=real_data)
  elif model_type == "stn":
    model = STAE(gamma=flags.gamma,
                 capacity_limit=flags.capacity_limit,
                 capacity_change_duration=flags.capacity_change_duration,
                 learning_rate=flags.learning_rate, flags=flags, real_data=real_data)
  else:
    raise ValueError("Model type is not defined: " + flags.model_type)
  print("Model type is " + flags.model_type)

  sess.run(tf.global_variables_initializer())

  saver = load_checkpoints(sess, model_path)

  features = []
  for shape in xrange(dims[0]):
    for scale in xrange(dims[1]):
      for orientation in xrange(dims[2]):
        batch_xs = []
        for i in xrange(dims[3]):
          for j in xrange(dims[4]):
            img = manager.get_image(
              shape=shape, scale=scale, orientation=orientation, x=i, y=j)
            batch_xs.append(img)
        z_mean, _ = model.transform(sess, batch_xs)
        index = 0
        for i in xrange(dims[3]):
          for j in xrange(dims[4]):
            feature = [shape, scale, orientation, i, j]
            feature.extend(z_mean[index])
            features.append(feature)
            index += 1
  tf.reset_default_graph()
  return features

def save_features(features, path):
  with open(path, 'w') as f:
    for feature in features:
      for x in feature:
        f.write(str(x) + "\t")
      f.write('\n')

def load_features(path):
  features = []
  with open(path, 'r') as f:
    for line in f:
      terms = line.split()
      feature = []
      for term in terms:
        feature.append(float(term))
      features.append(feature)
  return features

def split_features(features):
  x = []
  y = []
  for feature in features:
    x.append(feature[5:])
    y.append(feature[:5])
  return x, y

def binary_model():
  x = tf.placeholder(tf.float32, shape=(None, 20,), name='x')
  y = tf.placeholder(tf.int64, shape=(None,), name='y')

  W = tf.Variable(tf.zeros([20, 2]))
  b = tf.Variable(tf.zeros([2]))
  logits = tf.matmul(x, W) + b

  #logits = tf.layers.dense(x, 2)
  loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    labels=y, logits=logits))

  if flags.alpha != 0:
    loss += flags.alpha * tf.reduce_mean(tf.abs(W))

  prediction = tf.argmax(logits, 1)
  equality = tf.equal(prediction, y)
  accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))
  return x, y, loss, accuracy


def get_label(l_a, l_b, task_id):
  if task_id == 0: # is a left?
    return l_a[3] < l_b[3]
  elif task_id == 1: # is a down ?
    return l_a[4] < l_b[4]
  elif task_id == 2: # is a smaller?
    return l_a[1] < l_b[1]
  else:
    raise ValueError("Task ID is not defined: " + task_id)

def get_batch(labels, features, indice_a, indice_b, task_id):
  assert len(indice_a) == len(indice_b)
  x = []
  y = []
  for index_a, index_b in zip(indice_a, indice_b):
    x.append(features[index_a] + features[index_b])
    y.append(get_label(labels[index_a], labels[index_b], task_id))
  return x, y

def train(sess, source_manager, source_x, source_y, optimizer, x, y, loss, accuracy):
  sess.run(tf.global_variables_initializer())

  fetch = [optimizer, loss, accuracy]
  n_samples = len(source_y)
  for epoch in range(flags.epoch_size):
    # resample indice
    indices_a = source_manager.get_dependent_indices(n_samples)
    indices_b = source_manager.get_dependent_indices(n_samples)
    total_batch = n_samples // flags.batch_size

    avg_cost = 0.0
    avg_acc = 0.0
    for i in range(total_batch):
      batch_indices_a = indices_a[flags.batch_size*i : flags.batch_size*(i+1)]
      batch_indices_b = indices_b[flags.batch_size*i : flags.batch_size*(i+1)]
      batch_x, batch_y = get_batch(
        source_y,
        source_x,
        batch_indices_a,
        batch_indices_b,
        flags.task_id)
      feed = {x: batch_x, y: batch_y}
      _, out_loss, out_accuracy = sess.run(fetch, feed_dict=feed)
      avg_cost += out_loss
      avg_acc += out_accuracy
      if i % 100 == 0:
        print(i, avg_acc / 100., avg_cost / 100.)
        avg_cost = 0
        avg_acc = 0

def test(sess, target_manager, target_x, target_y, x, y, loss, accuracy):
  fetch = [loss, accuracy]
  n_samples = len(target_y)

  indices_a = target_manager.get_dependent_indices(n_samples)
  indices_b = target_manager.get_dependent_indices(n_samples)
  total_batch = n_samples // flags.batch_size

  avg_cost = 0.0
  avg_acc = 0.0
  for i in range(total_batch):
    batch_indices_a = indices_a[
                      flags.batch_size * i: flags.batch_size * (i + 1)]
    batch_indices_b = indices_b[
                      flags.batch_size * i: flags.batch_size * (i + 1)]
    batch_x, batch_y = get_batch(
      target_y,
      target_x,
      batch_indices_a,
      batch_indices_b,
      flags.task_id)
    feed = {x: batch_x, y: batch_y}
    out_loss, out_accuracy = sess.run(fetch, feed_dict=feed)
    avg_cost += out_loss
    avg_acc += out_accuracy
  print(avg_acc / total_batch, avg_cost / total_batch)

def main(argv):
  seed = flags.random_seed
  random.seed(seed)
  np.random.seed(seed)
  tf.set_random_seed(seed)

  if not os.path.isdir(my_path):
    os.makedirs(my_path)

  source_dims = [3, 6, 40, 32, 32]
  target_dims = [4, 4, 4, 15, 15]

  # source domain feature extraction
  print("\nExtracting source features.\n")
  source_manager = DataManager("customize", source_dims)

  source_path = my_path + "/source_features.txt"
  if os.path.exists(source_path):
    source_feartures = load_features(source_path)
  else:
    source_feartures = extract_features(
      source_manager,
      flags.model_type,
      flags.source_model,
      source_dims,
      False)
    save_features(source_feartures, source_path)
  source_x, source_y = split_features(source_feartures)
  print(len(source_feartures))

  # target domain feature extraction
  print("\nExtracting target features.\n")
  target_manager = RealImageGenerator("real_scustomize", target_dims)

  target_path = my_path + "/target_features.txt"
  if os.path.exists(target_path):
    target_feartures = load_features(target_path)
  else:
    target_feartures = extract_features(
      target_manager,
      flags.model_type,
      flags.target_model,
      target_dims,
      True)
    save_features(target_feartures, target_path)
  target_x, target_y = split_features(target_feartures)
  print(len(target_feartures))

  # prepare model
  x, y, loss, accuracy = binary_model()

  # train binary classifier in source domain
  optimizer = tf.train.AdamOptimizer(
    learning_rate=flags.learning_rate).minimize(loss)

  with tf.Session() as sess:
    # train in source domain
    print('\nTraining on Source Domain.\n')
    train(sess, source_manager, source_x, source_y, optimizer, x, y, loss,
          accuracy)

    # transfer test in target domain
    print('\nTesting on Target Domain.\n')
    test(sess, target_manager, target_x, target_y, x, y, loss, accuracy)

    print('\nTesting on Source Domain.\n')
    test(sess, source_manager, source_x, source_y, x, y, loss, accuracy)


if __name__ == '__main__':
  tf.app.run()
