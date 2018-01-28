import os
import numpy as np
import tensorflow as tf
from config import Config
from ops import tf_fun
from tqdm import tqdm


class data_processing(object):
    def __init__(self):
        self.name = 'crcns_2d_cell_5'
        self.config = Config()
        self.file_extension = '.csv'
        self.timepoints = 10  # Data is 100hz
        self.output_size = [1, 1]
        self.im_size = [self.timepoints, 256, 256, 1]
        self.model_input_image_size = [64, 64, 1]
        self.default_loss_function = 'pearson'
        self.score_metric = 'pearson'
        self.fix_imbalance = True
        self.preprocess = ['resize']
        self.train_prop = 0.80

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
            'image': tf_fun.bytes_feature,
            'label': tf_fun.float_feature
        }
        self.tf_dict = {
            'image': tf_fun.fixed_len_feature(dtype='string'),
            'label': tf_fun.fixed_len_feature(dtype='float')
        }
        self.tf_reader = {
            'image': {
                'dtype': tf.float16,
                'reshape': self.im_size
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
                'id': 5,
                'name': '20120417_cell5',
                'images': [
                    'cell5_002_001.npy',
                    'cell5_002_002.npy',
                    'cell5_002_003.npy',
                    'cell5_002_004.npy',
                    'cell5_002_005.npy',
                    'cell5_002_006.npy',
                    ],
                'e': [
                    'data_20120417_cell5_002_e.npy',
                ],
                'f': [
                    'data_20120417_cell5_002_f.npy',
                ],
                's': [
                    'data_20120417_cell5_002_s.npy',
                ],
            },
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
            for ims in tqdm(
                    d['images'],
                    total=len(d['images']),
                    desc='Images of cell %s' % d['id']):
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
            for e in tqdm(
                    d['e'],
                    total=len(d['e']),
                    desc='E of cell %s' % d['id']):
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
            for f in tqdm(
                    d['f'],
                    total=len(d['f']),
                    desc='F of cell %s' % d['id']):
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
            spike_array = np.zeros(len(time_f[d['id']]))
            for idx in tqdm(
                    range(1, len(time_f[d['id']])), desc='Binning spikes'):
                it_start = float(format(time_f[d['id']][idx - 1], '.4f'))
                it_end = float(format(time_f[d['id']][idx], '.4f'))
                e_start = form_e.index(it_start)
                if it_end not in form_e:
                    break
                else:
                    e_end = form_e.index(it_end)
                binned_spikes = np.sum(
                    [spikes[d['id']][x] for x in range(e_start, e_end)])
                spike_array[idx - 1] = binned_spikes
            import ipdb;ipdb.set_trace()
            selected_spikes[d['id']] = spike_array

            # Load s
            for s in tqdm(
                    d['s'],
                    total=len(d['s']),
                    desc='S of cell %s' % d['id']):
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
        cat_images = np.concatenate([v for v in images.values()])
        if len(cat_images.shape) == 3:
            cat_images = np.expand_dims(cat_images, axis=-1)
        cat_labels = np.concatenate([v for v in selected_spikes.values()])
        num_images = len(cat_images)
        num_labels = len(cat_labels)
        assert num_images == num_labels, 'Different numbers of ims/labs'

        # Cut into distinct events -- start with reshape for this
        num_events = int(num_images / self.timepoints)
        total_events = num_events * self.timepoints
        cat_images = cat_images[:total_events]
        cat_labels = cat_labels[:total_events]
        im_shape = cat_images.shape
        cat_images = cat_images.reshape(
            num_events,
            self.timepoints,
            im_shape[1],
            im_shape[2],
            im_shape[3])
        cat_labels = cat_labels.reshape(num_events, -1)

        # Split into train/test
        cat_labels = np.expand_dims(cat_labels.sum(-1), axis=-1)
        cv_split = np.round(num_events * self.train_prop).astype(int)
        train_images = cat_images[:cv_split]
        test_images = cat_images[cv_split:]
        train_labels = cat_labels[:cv_split]
        test_labels = cat_labels[cv_split:]

        # Fix imbalance with repetitions if requested
        if self.fix_imbalance:
            spike_idx = train_labels > 0
            num_spikes = spike_idx.sum()
            num_events = len(train_labels)
            num_neg = num_events - num_spikes
            rep = int(num_neg / num_spikes)
            spike_images = train_images[spike_idx.squeeze()]
            rep_spike_images = spike_images.repeat(rep, axis=0)
            rep_spike_labels = train_labels[spike_idx].repeat(rep, axis=0)
            train_images = np.concatenate(
                (train_images, rep_spike_images), axis=0)
            train_labels = np.concatenate(
                (train_labels, rep_spike_labels[:, None]), axis=0)

        # Sum labels per event (total spikes)
        files = {
            'train': train_images,
            'test': test_images
        }
        labels = {
            'train': train_labels,
            'test': test_labels
        }
        print 'Spikes in training: %s' % np.sum(cat_labels[:cv_split])
        print 'Spikes in testing: %s' % np.sum(cat_labels[cv_split:])
        return files, labels

    def get_data(self):
        files, labels = self.pull_data()
        return files, labels
