"""Create gabor filters to initialize model weights."""
import os
import numpy as np
from skimage.filters import gabor_kernel
from skimage.util import pad


out_name = 'gabors_for_contours_11.npy'
out_path = os.path.join(
    '%smedia' % os.path.sep,
    'data_cifs',
    'clicktionary',
    'pretrained_weights',
    out_name)

thetas = (np.arange(4.) / 4.) * np.pi
offsets = [0, 90]
scales_stds = [[0.2, 1.5]]  # [[0.2, 1]] [[0.2, 1]] [[0.2, 0.5]]
target_h = 11  # 7 5

filters = []
for ss in scales_stds:
    for offset in offsets:
        for theta in thetas:
            it_filter = np.real(
                    gabor_kernel(
                        ss[0],
                        theta=theta,
                        offset=offset,
                        n_stds=ss[1]))
            pad_h = (target_h - it_filter.shape[0]) / 2
            if pad_h:
                it_filter = pad(
                    it_filter, pad_h, 'constant', constant_values=0)
            print it_filter.shape
            filters += [it_filter]
# Paste the rectified versions together
rect_filters = np.asarray(filters)
# rect_filters = np.concatenate((np.minimum(rect_filters, 0), np.maximum(rect_filters, 0)), axis=0)
filters = rect_filters
filters = np.expand_dims(
    np.asarray(filters).transpose(1, 2, 0), axis=-2).astype(np.float32)
np.save(out_path, {'s1': [filters, []]})
print 'Created %s filters' % filters.shape[-1]

