import os
import numpy as np
from glob import glob
from tqdm import tqdm
from imageio import imread


image_path = os.path.join(
    '%smedia' % os.path.sep,
    'data_cifs',
    'image_datasets',
    'contours_gilbert_256_sparse_contrast')
images = glob(os.path.join(image_path, '*.png'))
ims = []
for im in tqdm(images, total=len(images)):
    ims += [imread(im)]
ims = np.asarray(ims)
ims = ims.reshape([len(images), -1])
mu = ims.mean(0)
sd = std(0)

