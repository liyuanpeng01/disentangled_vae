from data_manager import DataManager

from PIL import Image
import numpy as np

class RealImageGenerator(DataManager):
  def __init__(self, dist_type=None, dims=[32, 32]):
    super(RealImageGenerator, self).__init__(dist_type, dims)
    self.stride_v = 2
    self.stride_h = 2

    self.arr_v = 12
    self.arr_h = 12

    self.plate_v_half = 32
    self.plate_h_half = 32
    self.plate_v = 2 * self.plate_v_half
    self.plate_h = 2 * self.plate_h_half

    self.ori_imgs = []
    self.ori_imgs.append(self._load_image("data/transparent_data/keys.png"))
    self.ori_imgs.append(self._load_image("data/transparent_data/apple.png"))
    self.ori_imgs.append(self._load_image("data/transparent_data/spoon.png"))
    self.ori_imgs.append(self._load_image("data/transparent_data/mouse.png"))

  def load(self):
    latents_sizes = np.asanyarray([1, 4, 4, 4, 15, 15])
    # color, shape, scale, orientation, posX, posY

    self.n_samples = latents_sizes[::-1].cumprod()[-1]
    # 737280

    self.latents_bases = np.concatenate((latents_sizes[::-1].cumprod()[::-1][1:],
                                         np.array([1,])))

    self.imgs = []
    for shape in range(latents_sizes[1]):
      for scale in range(latents_sizes[2]):
        for orientation in range(latents_sizes[3]):
          for posX in range(latents_sizes[4]):
            for posY in range(latents_sizes[5]):
              self.imgs.append(self._prepare_image(shape, scale, orientation, posX, posY))

  def reshape_image(self, img):
      return img.reshape(12288)

  def _load_image(self, path):
    img = Image.open(path).convert("RGB")
    img.thumbnail((self.arr_v, self.arr_h), Image.ANTIALIAS)
    return img

  def _prepare_image(self, shape=0, scale=0, orientation=0, x=0, y=0):
    """
    Get a image with latent variables.
    :param shape: [1, 4]
    :param scaling: [1, 4]
    :param rotation: [1, 4]
    :param x: [1, 15]
    :param y: [1, 15]
    :return: a image
    """
    shape = shape
    scale = 0.5 * scale + 1
    orientation = 90 * orientation
    x = (x - 7) * self.stride_v
    y = (y - 7) * self.stride_h

    img = self.ori_imgs[shape]

    # rotation
    img = img.rotate(orientation)

    # scaling
    rescale_size = (
        int(round(scale * self.arr_v)),
        int(round(scale * self.arr_h)))
    img = img.resize(rescale_size, Image.ANTIALIAS)

    # translation
    arr = np.array(img)
    arr = np.divide(arr, 255.0)
    plate = np.zeros((self.plate_v, self.plate_h, 3))

    arr_v_half = len(arr) / 2
    arr_h_half = len(arr[0]) / 2
    v_start = self.plate_v_half - arr_v_half + y
    h_start = self.plate_h_half - arr_h_half + x
    for i in range(len(arr)):
      v = v_start + i
      if v < 0 or v >= len(plate):
        raise ValueError("v: " + v)
      for j in range(len(arr[0])):
        h = h_start + j
        if h < 0 or h >= len(plate[0]):
          raise ValueError("h: " + h)
        for k in range(len(arr[0][0])):
          plate[v][h][k] = arr[i][j][k]
    return plate

if __name__ == '__main__':
    ig = RealImageGenerator()
    ig.load()
    plate = ig.get_image(3, 3, 1, 10, 12)
    plate = plate.reshape((64, 64, 3))
    im = Image.fromarray(np.uint8(plate*255))
    im.show()
