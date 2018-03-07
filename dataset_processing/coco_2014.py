import os
import numpy as np
import tensorflow as tf
from glob import glob
from config import Config
from ops import tf_fun
import random


class data_processing(object):
    def __init__(self):
        self.name = 'coco_2014'
        self.aux_dir = 'coco_images'
        self.extension = '.jpg'
        self.config = Config()
        self.output_size = [89, 1]
        self.im_size = [256, 256, 3]
        self.model_input_image_size = [224, 224, 3]
        self.default_loss_function = 'sigmoid'
        self.image_meta_file = '_annotations.npy'
        self.score_metric = 'f1'
        self.preprocess = ['pad_resize']
        self.shuffle = False  # Preshuffle data?
        self.folds = {
            'train': 'train2014',
            'val': 'val2014'
        }
        self.targets = {
            'image': tf_fun.bytes_feature,
            'label': tf_fun.int32_feature
        }
        self.tf_dict = {
            'image': tf_fun.fixed_len_feature(dtype='string'),
            'label': tf_fun.fixed_len_feature(
                length=self.output_size[0],
                dtype='int32')
        }
        self.tf_reader = {
            'image': {
                'dtype': tf.float32,
                'reshape': self.im_size
            },
            'label': {
                'dtype': tf.int32,
                'reshape': None
            }
        }

    def get_data(self):
        files = self.get_files()
        labels = self.get_labels(files)
        return files, labels

    def get_files(self):
        files = {}
        for k, fold in self.folds.iteritems():
            it_files = glob(
                os.path.join(
                    self.config.data_root,
                    self.name,
                    self.aux_dir,
                    fold,
                    '*%s' % self.extension))
            if self.shuffle:
                random.shuffle(it_files)
            files[k] = it_files
        return files

    def get_labels(self, files):
        labels = {}
        for k, v in files.iteritems():
            meta_file = np.load(
                os.path.join(
                    self.config.data_root,
                    self.name,
                    self.aux_dir,
                    '%s%s' % (k, self.image_meta_file))).item()
            it_labels = []
            for idx, f in enumerate(v):
                try:
                    it_items = meta_file[f.split('/')[-1]]
                    it_labels_item = np.zeros(
                        (self.output_size[0]),
                        dtype=np.int64)
                    for il in it_items:
                        it_labels_item[il] = 1
                    it_labels += [list(it_labels_item)]
                except:
                    pass
            labels[k] = it_labels
        return labels
