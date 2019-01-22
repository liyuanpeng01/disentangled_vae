# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import math

class Distribution(object):
  def convert(self, v, r):
    boarder = v == 0 or v == r - 1
    value = (2. * v) / (r - 1.) - 1
    return value, boarder

  def get_scores(self, x):
    assert len(x) == 5
    max_border = 0
    scores = []
    for shape in range(x[0]):
      a, ba = self.convert(shape, x[0])
      for scale in range(x[1]):
        b, bb = self.convert(scale, x[1])
        for rotation in range(x[2]):
          c, bc = self.convert(rotation, x[2])
          for horizon in range(x[3]):
            d, bd = self.convert(horizon, x[3])
            for vertical in range(x[4]):
              e, be = self.convert(vertical, x[4])
              v = [a, b, c, d, e]
              score = self.get_score(v)
              scores.append(score)
              if bb or bc or bd or be:
                max_border = max(max_border, score)
    print("max boarder", max_border)
    scores = [max(0, score - max_border) for score in scores]

    count = 0
    for score in scores:
      if score > 0:
        count += 1
    print("valid values", count, len(scores), (100. * count) / len(scores))

    return scores

  def get_distribution(self, x):
    #x.reverse()
    scores = self.get_scores(x)
    ss = sum(scores)
    return [x / ss for x in scores]

  def get_scores1(self, valuables, values=[]):
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

class CustomizedDistribution(Distribution):
  def get_score(self, x):
    shape = x[0]
    #x = x[1:]
    x[0] *= 0.1
    score = 0
    for i in xrange(len(x)):
      for j in xrange(i, len(x)):
        s = x[i] * x[j]
        score += s
    #score *= (1 + 0.01 * shape)
    #score *= 2
    score = math.exp(-score)
    return score

class RealCustomizedDistribution(Distribution):
  def get_score(self, x):
    x[0] *= 0.1
    score = 0
    for i in xrange(len(x)):
      score += x[i] * x[i]
      for j in xrange(i, len(x)):
        score += x[i] * x[j]
    score = math.exp(-score)
    return score

class CustomizedDistribution_alpha(Distribution):
  def get_score(self, x):
    a = x[-1]
    b = x[-2]
    c = x[-3]
    d = x[-4]
    #score = a**2 + b**2 + a * b
    score = a ** 2 + b ** 2 + c ** 2 + a * b + a * c + b * c
    score += d **2 + d * (a + b + c)
    score = math.exp(-2 * score)
    return score

class CustomizedDistribution6(Distribution):
  def get_score(self, x):
    type = x[0]
    x = x[1:]
    s = sum(x)
    score = s * s
    score *= (4 + type)
    score = math.exp(-score)
    return score

class CustomizedDistribution5(Distribution):
  def get_score(self, x):
    type = x[0]
    x = x[1:]
    score = 0
    for i in xrange(len(x)):
      for j in xrange(i, len(x)):
        score += x[i] * x[j]
    score *= (type + 2)
    score = math.exp(-score)
    return score

class CustomizedDistribution1(Distribution):
  def get_score(self, x):
    th = sum([a * a for a in x])
    if th > 1:
      score = 0
    else:
      score = 1
    return score

class CustomizedDistribution3(Distribution):
  def get_score(self, x):
    th = sum([abs(a) for a in x])
    if th >= 1.5:
      score = 0
    else:
      score = 1
    return score

class CustomizedDistribution4(Distribution):
  def get_score(self, x):
    if sum(x) <= 0:
      score = 0
    else:
      score = 1
    return score

class CustomizedDistribution2(Distribution):
  def get_score(self, x):
    a = x[-2]
    b = x[-1]
    if abs(a) + abs(b) >= 1:
      score = 0
    else:
      score = 1
      for xi in x:
        score *= xi
      score = 1 - score
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
    elif dist_type == 'customize':
      dist = CustomizedDistribution()
      self.dist = dist.get_distribution(dims)
    elif dist_type == 'real_scustomize':
      dist = RealCustomizedDistribution()
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

  def get_index(self, latents):
    return np.dot(latents, self.latents_bases).astype(int)

  def get_image(self, shape=0, scale=0, orientation=0, x=0, y=0):
    latents = [0, shape, scale, orientation, x, y]
    index = self.get_index(latents)
    return self.get_images([index])[0]

  def reshape_image(self, img):
      return img.reshape(4096)

  def get_images(self, indices):
    images = []
    for index in indices:
      img = self.imgs[index]
      img = self.reshape_image(img)
      images.append(img)
    return images

  def get_dependent_indices(self, size):
    return np.random.choice(len(self.dist), size=size, p=self.dist)

  def get_random_images(self, size):
    indices = [np.random.randint(self.n_samples) for i in range(size)]
    return self.get_images(indices)
