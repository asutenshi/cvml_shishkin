# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.measure import regionprops, label
from collections import Counter
from itertools import combinations

# %%
data = Path('./task')

# %%
def concat_i(labels):
  props = regionprops(labels)
  new_labels = labels.copy()

  for prop1, prop2 in combinations(props, 2):
    top, bottom = (prop1, prop2) if prop1.bbox[0] < prop2.bbox[0] else (prop2, prop1)
    vertical_gap = bottom.bbox[0] - top.bbox[2]

    horizontal_diff = abs(top.centroid[1] - bottom.centroid[1])

    if 0 <= vertical_gap <= 20 and horizontal_diff < 10:
      new_labels[labels == prop2.label] = prop1.label
  return new_labels

# %%
def extractor(image):
  if image.ndim == 2:
    binary = image
  else:
    gray = (np.mean(image, axis=2) * 255).astype('uint8')
    binary = gray > 10
  labels = label(binary)
  new_labels = concat_i(labels)

  props = regionprops(new_labels)[0]

  return np.array([
    props.eccentricity,
    props.extent,
    props.euler_number,
    props.solidity,
    props.orientation,
    props.moments_hu[0],
    props.moments_hu[1],
    props.moments_hu[2],
    props.moments_hu[3],
    props.moments_hu[4],
    props.moments_hu[5],
    props.moments_hu[6],
  ], dtype='f4')

# %%
def make_train(path):
  train = []
  responses = []
  translate = {}
  ncls = 0
  for cls in sorted(path.glob('*')):
    ncls += 1
    translate[ncls] = cls.name if cls.name[0] != 's' else cls.name[1]
    for p in sorted(cls.glob('*.png')):
      train.append(extractor(plt.imread(p)))
      responses.append(ncls)
  train = np.array(train).reshape(-1, train[-1].shape[0])
  responses = np.array(responses, dtype='f4').reshape(-1, 1)
  return train, responses, translate

# %%
train, responses, translate = make_train(data / 'train')
knn = cv2.ml.KNearest.create()
knn.train(train, cv2.ml.ROW_SAMPLE, responses)

# %%
def need_space(propl, propr):
  # (min_row, min_col, max_row, max_col)
  xl = propl.bbox[3]
  xr = propr.bbox[1]
  diff = xr - xl
  if diff > 30: return True
  return False

# %%
phrases = []
phrases_true = [
  'C is LOW-LEVEL',
  'C++ is POWERFUL',
  'Python is INTUITIVE',
  'Rust is SAFE',
  'LUA is EASY',
  'Javascript is UGLY',
  'PHP sucks'
]

for i, image_path in enumerate(sorted(data.glob('*.png'))):
  image = plt.imread(image_path)

  gray = (np.mean(image, axis=2) * 255).astype('uint8')
  binary = gray > 0

  labels = label(binary)
  new_labels = concat_i(labels)
  props = regionprops(new_labels)
  props = sorted(props, key=lambda x: x.centroid[1])
  find = []
  for prop in props:
    find.append(extractor(prop.image))

  find = np.array(find, dtype='f4').reshape(-1, find[0].shape[0])

  ret, results, neighbours, dist = knn.findNearest(find, 5)
  phrase = ''

  prev_prop = props[-1]
  for res, prop in zip(results.flatten(), props):
    if need_space(prev_prop, prop):
      phrase += ' ' + translate[res]
    else:
      phrase += translate[res]
    prev_prop = prop
  phrases.append(phrase)

  print(i, phrase)

# c_true = 0
# c_false = 0

# for pred, lb in zip(phrases, phrases_true):
#   for ch1, ch2 in zip(pred, lb):
#     if ch1 == ch2: c_true += 1
#     else: c_false += 1
# accuracy = ~91%
# print(f'Accuracy = {c_true / (c_true + c_false)}')
