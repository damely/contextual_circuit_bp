import os
import re
import numpy as np
import tensorflow as tf
from glob import glob
from config import Config
from ops import tf_fun


class data_processing(object):
    def __init__(self):
        self.name = 'contours_gilbert_256_contrast'
        self.im_extension = '.png'
        self.images_dir = 'images'
        self.label_regex = r'(?<=length)\d+'
        self.config = Config()
        self.im_size = [256, 256, 3]  # 600, 600
        self.model_input_image_size = [256, 256, 3]  # [107, 160, 3]
        self.max_ims = 0
        self.output_size = [1]
        self.label_size = self.output_size
        self.default_loss_function = 'cce'
        self.score_metric = 'accuracy'
        self.store_z = False
        self.normalize_im = True
        self.shuffle = True
        self.input_normalization = 'none'
        self.preprocess = ['resize']  # ['resize_nn']
        self.folds = {
            'train': 'train',
            'val': 'val'
        }
        self.cv_split = 0.9
        self.cv_balance = True
        self.targets = {
            'image': tf_fun.bytes_feature,
            'label': tf_fun.int64_feature
        }
        self.tf_dict = {
            'image': tf_fun.fixed_len_feature(dtype='string'),
            'label': tf_fun.fixed_len_feature(dtype='int64')
        }
        self.tf_reader = {
            'image': {
                'dtype': tf.float32,
                'reshape': self.im_size
            },
            'label': {
                'dtype': tf.int64,
                'reshape': self.output_size
            }
        }

    def get_data(self):
        """Get the names of files."""
        files = np.asarray(
            glob(
                os.path.join(
                    self.config.data_root,
                    self.name,
                    '*%s' % self.im_extension)))
        labels = np.asarray(
            [int(re.search(self.label_regex, x).group()) for x in files])
        labels = (labels > 1).astype(np.int32)
        ul, lc = np.unique(labels, return_counts=True)
        include_count = np.min(lc)
        if self.max_ims:
            include_count = np.min([include_count, lc])

        # Trim files and labels to include_count
        pos_idx = np.where(labels == 1)[0][:include_count]
        neg_idx = np.where(labels == 0)[0][:include_count]

        # Create CV folds
        cv_files, cv_labels = {}, {}
        cv_files[self.folds['train']] = {}
        cv_files[self.folds['val']] = {}
        prev_cv = 0
        for k, v in self.folds.iteritems():
            if k == self.folds['train']:
                cv_split = int(include_count * self.cv_split)
            elif k == self.folds['val']:
                cv_split = int(include_count * (1 - self.cv_split))
            else:
                raise NotImplementedError
            if prev_cv:
                cv_split += prev_cv
            cv_inds = np.arange(prev_cv, cv_split)
            it_files = np.concatenate((
                files[pos_idx][cv_inds],
                files[neg_idx][cv_inds]))
            it_labels = np.concatenate((
                labels[pos_idx][cv_inds],
                labels[neg_idx][cv_inds]))
            if self.shuffle:
                shuffle_idx = np.random.permutation(len(it_files))
                it_files = it_files[shuffle_idx]
                it_labels = it_labels[shuffle_idx]
            cv_files[k] = it_files
            cv_labels[k] = it_labels
            prev_cv = cv_split
        return cv_files, cv_labels

