import os
import numpy as np
import tensorflow as tf
from glob import glob
from config import Config
from ops import tf_fun
from utils import py_utils
from scipy import io, misc


class data_processing(object):
    def __init__(self):
        self.name = 'BSDS500_2'
        self.orig_name = 'BSDS500'
        self.im_extension = '.jpg'
        self.lab_extension = '.mat'
        self.images_dir = 'images'
        self.labels_dir = 'groundTruth'
        self.processed_labels = 'processed_labels'
        self.processed_images = 'processed_images'
        self.config = Config()
        self.im_size = [321, 481, 3]
        self.model_input_image_size = [107, 160, 3]  # [150, 240, 3]
        self.output_size = [321, 481, 1]  # 256 x 256 pixels, 7 x 7 angle
        self.label_size = self.output_size
        self.default_loss_function = 'pearson'
        self.score_metric = 'pearson'
        self.aux_scores = ['f1']
        self.preprocess = [None]  # ['resize_nn']
        self.folds = {
            'train': 'train',
            'val': 'val'
        }
        self.fold_options = {
            'train': 'mean',
            'val': 'duplicate'
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
        files = self.get_files()
        labels, files = self.get_labels(files)

        return files, labels

    def get_files(self):
        """Get the names of files."""
        files = {}
        for k, fold in self.folds.iteritems():
            it_files = glob(
                os.path.join(
                    self.config.data_root,
                    self.orig_name,
                    self.images_dir,
                    fold,
                    '*%s' % self.im_extension))
            files[k] = it_files
        return files

    def get_labels(self, files):
        """Process and save label images."""
        labels = {}
        new_files = {}
        for k, images in files.iteritems():
            # Replace extension and path with labels
            label_vec = []
            file_vec = []
            fold = images[0].split(os.path.sep)[-2]

            # New label dir
            proc_dir = os.path.join(
                images[0].split(fold)[0],
                fold,
                self.processed_labels)
            py_utils.make_dir(proc_dir)

            # New image dir
            proc_image_dir = os.path.join(
                self.config.data_root,
                self.name,
                self.images_dir,
                fold,
                self.processed_images)
            py_utils.make_dir(proc_image_dir)
            all_images = []
            for im in images:
                it_label = im.split(os.path.sep)[-1]
                it_label_path = '%s%s' % (im.split('.')[0], self.lab_extension)
                it_label_path = it_label_path.replace(
                    self.images_dir,
                    self.labels_dir)

                # Process every label and duplicate images for each
                label_data = io.loadmat(
                    it_label_path)['groundTruth'].reshape(-1)
                im_data = misc.imread(im)
                transpose_labels = False
                if not np.all(self.im_size == list(im_data.shape)):
                    im_data = np.swapaxes(im_data, 0, 1)
                    transpose_labels = True
                assert np.all(
                    self.im_size == list(im_data.shape)
                    ), 'Mismatched dimensions.'

                if self.fold_options[k] == 'duplicate':
                    # Loop through all labels
                    for idx, lab in enumerate(label_data):

                        # Process labels
                        ip_lab = lab.item()[1].astype(np.float32)
                        if transpose_labels:
                            ip_lab = np.swapaxes(ip_lab, 0, 1)
                        it_im_name = '%s_%s' % (idx, it_label)
                        it_lab_name = '%s.npy' % it_im_name.split('.')[0]
                        out_lab = os.path.join(proc_dir, it_lab_name)
                        np.save(out_lab, ip_lab)
                        label_vec += [out_lab]

                        # Process images
                        proc_im = os.path.join(proc_image_dir, it_im_name)
                        all_images.append(im_data)
                        file_vec += [proc_im]
                elif self.fold_options[k] == 'mean':
                    mean_labs = []
                    for idx, lab in enumerate(label_data):
                        # Process labels
                        ip_lab = lab.item()[1].astype(np.float32)
                        if transpose_labels:
                            ip_lab = np.swapaxes(ip_lab, 0, 1)
                        mean_labs += [ip_lab]
                    mean_lab = np.asarray(mean_labs).mean(0)
                    out_lab = os.path.join(
                        proc_dir, '%s.npy' % it_label.strip(
                            self.im_extension)[0])
                    np.save(out_lab, mean_lab)
                    label_vec += [out_lab]

                    # Process images
                    proc_im = os.path.join(proc_image_dir, it_label)
                    all_images.append(im_data)
                    file_vec += [proc_im]
                else:
                    raise NotImplementedError

            # Zscore the images.
            # load all images from filevec into an asarray
            # calculate the mean
            # calculate the std
            # zscore
            # Overwrite the saved images... use floating point flag on imsave
            # Alternatively, save with scikit-image
            all_images = np.array(all_images).astype(np.float32)
            if k == 'train':
                mean_rgb_train = all_images.mean(
                    0, keepdims=True).mean(
                    1, keepdims=True).mean(
                    2, keepdims=True)
                std_rgb_train = all_images.std()
            all_images_z = (all_images - mean_rgb_train) / std_rgb_train
            for i, img in enumerate(all_images_z):
                np.save(
                    '%s.npy' % file_vec[i].strip(self.im_extension),
                    img.astype(np.float32))
            # labels[k] = label_vec
            # new_files[k] = ['%s.npy' % i for i in file_vec]
            labels[k] = label_vec
            new_files[k] = ['%s.npy' % i for i in file_vec]
        return labels, new_files
