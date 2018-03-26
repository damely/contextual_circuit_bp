import os
import numpy as np
from glob import glob
from tqdm import tqdm
from imageio import imread
from skimage import color
from multiprocessing import Pool


class Processor:
    def __init__(self):
        pass

    def __call__(self,filename):
        return color.rgb2gray(imread(filename))


max_ims = 100000
processes = 8
image_path = os.path.join(
    '%smedia' % os.path.sep,
    'data_cifs',
    'image_datasets',
    'contours_gilbert_256_sparse_nonRandomShear')
images = glob(os.path.join(image_path, '*.png'))
if max_ims:
    images = images[:max_ims]
# ims = []
# for im in tqdm(images, total=len(images)):
#     ims += [color.rgb2gray(imread(im))]

proc = Processor()
p = Pool(processes=processes)
ims = p.map(proc, images)
# ims = np.empty([max_ims, 256, 256])
# for idx, im in pool.imap(imloader, images):
#     ims[idx, :, :] = im

ims = np.asarray(ims)
res_ims = ims.reshape(-1)
mu = res_ims.mean()
sd = res_ims.std()
np.savez('mu_sd', images={'mean': mu, 'std': sd})

# zims = (ims - mu) / (sd + 1e-12)

