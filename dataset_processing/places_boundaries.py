import os
import tensorflow as tf
from config import Config
from ops import tf_fun
import pandas as pd


class data_processing(object):
    def __init__(self):
        self.name = 'places_boundaries'
        self.im_extension = '.jpg'
        self.lab_extension = '.mat.npy'
        self.images_dir = 'images'
        self.labels_dir = 'groundTruth'
        self.processed_labels = 'processed_labels'
        self.processed_images = 'processed_images'
        self.config = Config()
        self.im_size = [321, 481, 3]
        self.model_input_image_size = [300, 480, 3]  # [107, 160, 3]
        self.output_size = [321, 481, 1]
        self.label_size = self.output_size
        self.default_loss_function = 'pearson'
        self.score_metric = 'pearson'
        self.aux_scores = ['f1']
        self.store_z = True        
        self.preprocess = [None]  # ['resize_nn']
        self.folds = {
            'train': 'training.txt',
            'val': 'validation.txt'
        }
        self.fold_options = {
            'train': 'mean',
            'val': 'mean'
        }
        self.targets = {
            'image': tf_fun.bytes_feature,
            'label': tf_fun.bytes_feature
        }
        self.tf_dict = {
            'image': tf_fun.fixed_len_feature(dtype='string'),
            'label': tf_fun.fixed_len_feature(dtype='string')
        }
        self.tf_reader = {
            'image': {
                'dtype': tf.float32,
                'reshape': self.im_size
            },
            'label': {
                'dtype': tf.float32,
                'reshape': self.output_size
            }
        }

    def get_data(self):
        """Get the names of files."""
        image_files, label_files = {}, {}
        for k, fold in self.folds.iteritems():
            im_list = pd.read_csv(
                os.path.join(
                    self.config.data_root,
                    self.name,
                    fold)).as_matrix().ravel().tolist()
            image_files[k] = [os.path.join(
                self.config.data_root,
                self.name,
                'images',
                x) for x in im_list]
            label_files[k] = [os.path.join(
                self.config.data_root,
                self.name,
                'labels',
                '%s%s' % (x.strip(self.im_extension), self.lab_extension))
                for x in im_list]
        return image_files, label_files
