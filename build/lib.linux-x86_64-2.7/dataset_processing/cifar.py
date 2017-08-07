import os
import re
from glob import glob
from config import Config
from ops import tf_fun 


class data_processing(object):
    def __init__(self):
        self.name = 'cifar'
        self.extension = '.png'
        self.config = Config()
        self.folds = {
            'train': 'train',
            'test': 'test'
        }
        self.targets = {
            'image': tf_fun.bytes_feature,
            'label': tf_fun.int64_feature
        }
        self.im_size = [32, 32, 3]

    def get_data(self):
        files = self.get_files()
        labels = self.get_labels(files)
        return files, labels

    def get_files(self):
        files = {}
        for k, fold in self.folds.iteritems():
            files[k] = glob(
                os.path.join(
                    self.config.data_root,
                    self.name,
                    fold,
                    '*%s' % self.extension))
        return files

    def get_labels(self, files):
        labels = {}
        for k, v in files.iteritems():
            it_labels = []
            for f in v:
                it_labels += [int(re.split('\.', f.split('_')[-1])[0])]
            labels[k] = it_labels
        return labels
