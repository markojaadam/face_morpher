from builtins import range
import cv2
import numpy as np
import scipy.sparse

def mask_from_points(size, points):
  radius = 10  # kernel size
  kernel = np.ones((radius, radius), np.uint8)

  mask = np.zeros(size, np.uint8)
  cv2.fillConvexPoly(mask, cv2.convexHull(points), 255)
  mask = cv2.erode(mask, kernel)

  return mask

def apply_mask(img, mask):
  """ Apply mask to supplied image
  :param img: max 3 channel image
  :param mask: [0-255] values in mask
  :returns: new image with mask applied
  """
  masked_img = np.copy(img)
  num_channels = 3
  for c in range(num_channels):
    masked_img[..., c] = img[..., c] * (mask / 255)

  return masked_img

def weighted_average(img1, img2, percent=0.5):
  if percent <= 0:
    return img2
  elif percent >= 1:
    return img1
  else:
    return cv2.addWeighted(img1, percent, img2, 1-percent, 0)

def alpha_feathering(src_img, dest_img, img_mask, blur_radius=15):
  mask = cv2.blur(img_mask, (blur_radius, blur_radius))
  mask = mask / 255.0

  result_img = np.empty(src_img.shape, np.uint8)
  for i in range(3):
    result_img[..., i] = src_img[..., i] * mask + dest_img[..., i] * (1-mask)

  return result_img

def poisson_blend(img_source, dest_img, img_mask, offset=(0, 0)):
  # http://opencv.jp/opencv2-x-samples/poisson-blending
  img_target = np.copy(dest_img)
  import pyamg
  # compute regions to be blended
  region_source = (
    max(-offset[0], 0),
    max(-offset[1], 0),
    min(img_target.shape[0] - offset[0], img_source.shape[0]),
    min(img_target.shape[1] - offset[1], img_source.shape[1]))
  region_target = (
    max(offset[0], 0),
    max(offset[1], 0),
    min(img_target.shape[0], img_source.shape[0] + offset[0]),
    min(img_target.shape[1], img_source.shape[1] + offset[1]))
  region_size = (region_source[2] - region_source[0],
                 region_source[3] - region_source[1])

  # clip and normalize mask image
  img_mask = img_mask[region_source[0]:region_source[2],
                      region_source[1]:region_source[3]]

  # create coefficient matrix
  coff_mat = scipy.sparse.identity(np.prod(region_size), format='lil')
  for y in range(region_size[0]):
    for x in range(region_size[1]):
      if img_mask[y, x]:
        index = x + y * region_size[1]
        coff_mat[index, index] = 4
        if index + 1 < np.prod(region_size):
          coff_mat[index, index + 1] = -1
        if index - 1 >= 0:
          coff_mat[index, index - 1] = -1
        if index + region_size[1] < np.prod(region_size):
          coff_mat[index, index + region_size[1]] = -1
        if index - region_size[1] >= 0:
          coff_mat[index, index - region_size[1]] = -1
  coff_mat = coff_mat.tocsr()

  # create poisson matrix for b
  poisson_mat = pyamg.gallery.poisson(img_mask.shape)
  # for each layer (ex. RGB)
  for num_layer in range(img_target.shape[2]):
    # get subimages
    t = img_target[region_target[0]:region_target[2],
                   region_target[1]:region_target[3], num_layer]
    s = img_source[region_source[0]:region_source[2],
                   region_source[1]:region_source[3], num_layer]
    t = t.flatten()
    s = s.flatten()

    # create b
    b = poisson_mat * s
    for y in range(region_size[0]):
      for x in range(region_size[1]):
        if not img_mask[y, x]:
          index = x + y * region_size[1]
          b[index] = t[index]

    # solve Ax = b
    x = pyamg.solve(coff_mat, b, verb=False, tol=1e-10)

    # assign x to target image
    x = np.reshape(x, region_size)
    x[x > 255] = 255
    x[x < 0] = 0
    x = np.array(x, img_target.dtype)
    img_target[region_target[0]:region_target[2],
               region_target[1]:region_target[3], num_layer] = x

  return img_target


def add_background(background_img, overlay_t_img):
  # Resize the canvas of overlay image
  bg_h, bg_w = background_img.shape[:2]
  pic_h, pic_w = overlay_t_img.shape[:2]
  overlay_t_img = cv2.copyMakeBorder(overlay_t_img,
                                     (bg_h - pic_h) / 2, (bg_h - pic_h) / 2, (bg_w - pic_w) / 2, (bg_w - pic_w) / 2,
                                     cv2.BORDER_CONSTANT)

  # Split out the transparency mask from the colour info
  overlay_img = overlay_t_img[:, :, :3]  # Grab the BRG planes
  overlay_mask = overlay_t_img[:, :, 3:]  # And the alpha plane

  # Again calculate the inverse mask
  background_mask = 255 - overlay_mask

  # Turn the masks into three channel, so we can use them as weights
  overlay_mask = cv2.cvtColor(overlay_mask, cv2.COLOR_GRAY2BGR)
  background_mask = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR)

  # Create a masked out face image, and masked out overlay
  # We convert the images to floating point in range 0.0 - 1.0
  face_part = (background_img * (1 / 255.0)) * (background_mask * (1 / 255.0))
  overlay_part = (overlay_img * (1 / 255.0)) * (overlay_mask * (1 / 255.0))

  # And finally just add them together, and rescale it back to an 8bit integer image
  return np.uint8(cv2.addWeighted(face_part, 255.0, overlay_part, 255.0, 0.0))
