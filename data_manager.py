# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import math

class Distribution(object):
  def get_distribution(self, x):
    x.reverse()
    scores = self.get_scores(x)
    ss = sum(scores)
    return [x / ss for x in scores]

  def get_scores(self, valuables, values=[]):
    if len(valuables) == 0:
      return [self.get_score(values)]
    dist = []
    r = valuables.pop()
    values.append(1)
    if r == 1:
      dist.extend(self.get_scores(valuables, values))
    else:
      for i in xrange(r):
        values[-1] = (2. * i) / (r - 1.) - 1
        dist.extend(self.get_scores(valuables, values))
    values.pop()
    valuables.append(r)
    return dist

  def get_score(self, x):
    pass

class LinearDistribution(Distribution):
  def get_score(self, x):
    score = 1
    for xi in x:
      score *= xi
    score = 1 - score
    return score

class GaussianDistribution(Distribution):
  def get_score(self, x):
    score = 0
    for i in xrange(len(x)):
      for j in xrange(i, len(x)):
        score += x[i] * x[j]
    score = math.exp(-score)
    return score

class DataManager(object):
  def __init__(self, dist_type=None, dims=[32, 32]):
    n = 32.
    ni = int(n)
    if dist_type is None:
      self.dist = []
      for i in xrange(ni):
        for j in xrange(ni):
          self.dist.append(0.25 * n + (n - i) + (1 + i - (n - i)) * j / n)
      s = sum(self.dist)
      for i in xrange(len(self.dist)):
        self.dist[i] /= s
    elif dist_type == 'linear':
      dist = LinearDistribution()
      self.dist = dist.get_distribution(dims)
    elif dist_type == 'gaussian':
      dist = GaussianDistribution()
      self.dist = dist.get_distribution(dims)
    else:
      with open(dist_type, 'r') as f:
        self.dist = [float(x) for x in f.readlines()]
    assert np.isclose(sum(self.dist), 1.)

  def load(self):
    # Load dataset
    dataset_zip = np.load('data/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz',
                          encoding = 'latin1')

    # print('Keys in the dataset:', dataset_zip.keys())
    #  ['metadata', 'imgs', 'latents_classes', 'latents_values']

    self.imgs       = dataset_zip['imgs']
    latents_values  = dataset_zip['latents_values']
    latents_classes = dataset_zip['latents_classes']
    metadata        = dataset_zip['metadata'][()]

    # Define number of values per latents and functions to convert to indices
    latents_sizes = metadata['latents_sizes']
    # [ 1,  3,  6, 40, 32, 32]
    # color, shape, scale, orientation, posX, posY

    self.n_samples = latents_sizes[::-1].cumprod()[-1]
    # 737280

    self.latents_bases = np.concatenate((latents_sizes[::-1].cumprod()[::-1][1:],
                                         np.array([1,])))
    # [737280, 245760, 40960, 1024, 32, 1]
    
  @property
  def sample_size(self):
    return self.n_samples

  def get_image(self, shape=0, scale=0, orientation=0, x=0, y=0):
    latents = [0, shape, scale, orientation, x, y]
    index = np.dot(latents, self.latents_bases).astype(int)
    return self.get_images([index])[0]

  def get_images(self, indices):
    images = []
    for index in indices:
      img = self.imgs[index]
      img = img.reshape(4096)
      images.append(img)
    return images

  def get_dependent_indices(self, size):
    return np.random.choice(len(self.dist), size=size, p=self.dist)

  def get_random_images(self, size):
    indices = [np.random.randint(self.n_samples) for i in range(size)]
    return self.get_images(indices)
