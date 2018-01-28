import os
import numpy as np
import tensorflow as tf
from config import Config
from ops import tf_fun


class data_processing(object):
    def __init__(self):
        self.name = 'spikefinder'
        self.config = Config()
        self.file_extension = '.csv'
        self.timepoints = 5  # Data is 100hz
        self.trim_nans = True
        self.output_size = [1, self.timepoints]
        self.im_size = [1, self.timepoints]
        self.model_input_image_size = [1, self.timepoints]
        self.default_loss_function = 'l2'
        self.score_metric = 'l2'
        self.preprocess = [None]

        # CRCNS data pointers
        self.crcns_dataset = os.path.join(
            self.config.data_root,
            'crcns',
            'cai-1')
        self.exp_name = 'GCaMP6s_9cells_Chen2013'

        # CC-BP dataset vars
        self.folds = {
            'train': 'train',
            'test': 'test'}
        self.targets = {
            'image': tf_fun.float_feature,
            'label': tf_fun.float_feature
        }
        self.tf_dict = {
            'image': tf_fun.fixed_len_feature(dtype='float'),
            'label': tf_fun.fixed_len_feature(dtype='float')
        }
        self.tf_reader = {
            'image': {
                'dtype': tf.float32,
                'reshape': None
            },
            'label': {
                'dtype': tf.float32,
                'reshape': None
            }
        }

    def pull_data(self):
        img_dir = os.path.join(
            self.crcns_dataset,
            self.exp_name)
        cell_data = [
            {
                'id': 1,
                'name': '20120416_cell1',
                'images': [
                    'cell1_001_001.npy',
                    'cell1_001_002.npy',
                    'cell1_001_003.npy',
                    'cell1_001_004.npy',
                    'cell1_001_005.npy',
                    'cell1_001_006.npy',
                    ],
                'e': [
                    'data_20120416_cell1_001_e.npy',
                ],
                'f': [
                    'data_20120416_cell1_001_f.npy',
                ],
                's': [
                    'data_20120416_cell1_001_s.npy',
                ],
            },
            {
                'id': 2,
                'name': '20120416_cell1',
                'images': [
                    'cell1_002_001.npy',
                    'cell1_002_002.npy',
                    'cell1_002_003.npy',
                    'cell1_002_004.npy',
                    'cell1_002_005.npy'
                    ],
                'e': [
                    'data_20120416_cell1_002_e.npy',
                    'data_20120417_cell1_002_e.npy'
                ],
                'f': [
                    'data_20120416_cell1_002_f.npy',
                    'data_20120417_cell1_002_f.npy'
                ],
                's': [
                    'data_20120416_cell1_002_s.npy',
                    'data_20120417_cell1_002_s.npy'
                ],
            }
        ]

        images, ss = {}, {}
        spikes, time_e = {}, {}
        corr_f, raw_f, time_f = {}, {}, {}
        selected_spikes = {}
        for d in cell_data:
            images[d['id']] = []
            spikes[d['id']] = []
            time_e[d['id']] = []
            corr_f[d['id']] = []
            raw_f[d['id']] = []
            time_f[d['id']] = []
            ss[d['id']] = []

            # Load images
            for ims in d['images']:
                images[d['id']] += [np.load(
                    os.path.join(
                        img_dir,
                        'raw_data',
                        'cell_npy',
                        d['name'],
                        ims)
                    )]
            images[d['id']] = np.concatenate(images[d['id']], axis=0)

            # Load e
            for e in d['e']:
                it_data = np.load(os.path.join(
                    img_dir,
                    'processed_data',
                    'npys',
                    'e',
                    e))
                it_data = np.asarray(
                    [list(x) for x in it_data])
                spikes[d['id']] += [it_data[:, 1]]
                time_e[d['id']] += [it_data[:, 0]]
            spikes[d['id']] = np.concatenate(spikes[d['id']]).squeeze()
            time_e[d['id']] = np.concatenate(time_e[d['id']]).squeeze()

            # Load f
            for f in d['f']:
                it_data = np.load(
                    os.path.join(
                        img_dir,
                        'processed_data',
                        'npys',
                        'f',
                        f)
                    )
                it_data = np.asarray(
                    [list(x) for x in it_data])
                corr_f[d['id']] += [it_data[:, 3]]
                raw_f[d['id']] += [it_data[:, 1]]
                time_f[d['id']] += [it_data[:, 0]]
            corr_f[d['id']] = np.concatenate(corr_f[d['id']]).squeeze()
            raw_f[d['id']] = np.concatenate(raw_f[d['id']]).squeeze()
            time_f[d['id']] = np.concatenate(time_f[d['id']]).squeeze()
            time_f[d['id']] = np.asarray([
                float(format(x, '.4f')) for x in time_f[d['id']]]).squeeze()
            selected_spikes[d['id']] = []
            form_e = [float(format(f, '.4f')) for f in time_e[d['id']]]
            for idx in range(1, len(time_f[d['id']])):
                it_start = float(format(time_f[d['id']][idx - 1], '.4f'))
                it_end = float(format(time_f[d['id']][idx], '.4f'))
                e_start = form_e.index(it_start)
                e_end = form_e.index(it_end)
                binned_spikes = np.sum(
                    [spikes[d['id']][x] for x in range(e_start, e_end)])
                selected_spikes[d['id']] += [binned_spikes]
            selected_spikes[d['id']] = np.concatenate(
                ([0], np.asarray(selected_spikes[d['id']]).squeeze()))

            # Load s
            for s in d['s']:
                ss[d['id']] += [np.load(
                    os.path.join(
                        img_dir,
                        'processed_data',
                        'npys',
                        's',
                        s)
                    )]
        # For each cell being processed:
        #    files = images
        #    labels = selected_spikes
        import ipdb;ipdb.set_trace()
        files = images
        labels = selected_spikes
        return files, labels

    def get_data(self):
        files, labels = self.pull_data()
        return files, labels
