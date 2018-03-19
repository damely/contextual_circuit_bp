import os
import re
import itertools
import numpy as np
from glob import glob
from skimage import transform, io
from tqdm import tqdm
from joblib import Parallel, delayed


def make_dir(d):
    """Make directory d if it does not exist."""
    if not os.path.exists(d):
        os.makedirs(d)


def create_rotations(combos, order, im_ext='.png'):
    """Create SKimage rotated images."""
    for combo in combos:
        f, theta = combo
        im = io.imread(f)
        transformed_im = transform.rotate(im, theta, order=order)
        f_path = f.split(os.path.sep)[-1]
        out_name = os.path.join(out_dir, f_path)
        out_name = '%s_rotation%s%s' % (
            out_name.strip(im_ext),
            theta,
            im_ext)
        io.imsave(out_name, transformed_im)


def joblib_loop(combos, order):
    Parallel(n_jobs=4)(
        delayed(
            create_rotations)(combos, order) for theta in thetas)


# Balance +/- and trim to max_ims total
max_ims = 10000
thetas = np.arange(5, 360, 5)
per_class = max_ims // 2
im_wc = '*.png'
label_regex = r'(?<=length)\d+'
order = 3  # bicubic

# Directories
all_ims = os.path.join(
    '%smedia' % os.path.sep,
    'data_cifs',
    'image_datasets',
    'contours_gilbert_256_length_0',
    im_wc)
out_dir = os.path.join(
    '%smedia' % os.path.sep,
    'data_cifs',
    'image_datasets',
    'contours_gilbert_256_length_0_rotated')
make_dir(out_dir)

# Paste the +/- class lists together
all_ims = np.asarray(glob(all_ims))
labels = np.asarray([int(re.search(label_regex, x).group()) for x in all_ims])
# num_pos = np.sum(labels == 1)
# num_neg = np.sum(labels == 0)
# trimmed_num = np.min([num_pos, num_neg])
pos_ims = np.where(labels == 1)[0][:per_class]
neg_ims = np.where(labels == 0)[0][:per_class]
trimmed_list = np.concatenate((all_ims[pos_ims], all_ims[neg_ims]))
combos = [[x, y] for x in trimmed_list for y in thetas]
%timeit joblib_loop(combos, order)
# create_rotations(trimmed_list, order, 5)
