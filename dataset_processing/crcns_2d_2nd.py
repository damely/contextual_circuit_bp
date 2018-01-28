import os
import numpy as np
import tensorflow as tf
from config import Config
from ops import tf_fun
from tqdm import tqdm


class data_processing(object):
    def __init__(self):
        self.name = 'crcns_2d_2nd'
        self.config = Config()
        self.file_extension = '.csv'
        self.timepoints = 10  # Data is 100hz
        self.output_size = [2]
        self.im_size = [self.timepoints, 256, 256, 1]
        self.model_input_image_size = [164, 164, 1]
        self.default_loss_function = 'sigmoid_logits'
        self.score_metric = 'argmax_softmax_accuracy'
        self.fix_imbalance_train = True
        self.fix_imbalance_test = False
        self.preprocess = ['resize']
        self.train_prop = 0.80
        self.binarize_spikes = True
        self.df_f_window = 10
        self.use_df_f = False
        self.shuffle_train = True
        self.shuffle_test = False
        self.overwrite_numpy = False
        self.validation_slice = 2

        # CRCNS data pointers
        self.crcns_dataset = os.path.join(
            self.config.data_root,
            'crcns',
            'cai-1')
        self.exp_name = 'GCaMP6s_9cells_Chen2013'
        self.numpy = os.path.join(
            self.crcns_dataset,
            '%s.npz' % self.exp_name)

        # CC-BP dataset vars
        self.folds = {
            'train': 'train',
            'test': 'test'
        }
        self.targets = {
            'image': tf_fun.bytes_feature,
            'label': tf_fun.int64_feature
        }
        self.tf_dict = {
            'image': tf_fun.fixed_len_feature(dtype='string'),
            'label': tf_fun.fixed_len_feature(
                length=self.output_size[-1], dtype='int64')
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

    def pull_data(self):
        img_dir = os.path.join(
            self.crcns_dataset,
            self.exp_name)
        cell_data = [
            {
                'id': 0,
                'cell_id': 0,
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
            # {
            #     'id': 1,
            #     'cell_id': 0,
            #     'name': '20120416_cell1',
            #     'images': [
            #         'cell1_002_001.npy',
            #         'cell1_002_002.npy',
            #         'cell1_002_003.npy',
            #         'cell1_002_004.npy',
            #         'cell1_002_005.npy',
            #         # 'cell1_002_006.npy',
            #         ],
            #     'e': [
            #         'data_20120416_cell1_002_e.npy',
            #     ],
            #     'f': [
            #         'data_20120416_cell1_002_f.npy',
            #     ],
            #     's': [
            #         'data_20120416_cell1_002_s.npy',
            #     ],
            # },
            {
                'id': 2,
                'cell_id': 0,
                'name': '20120417_cell1',
                'images': [
                    'cell1_002_001.npy',
                    'cell1_002_002.npy',
                    'cell1_002_003.npy',
                    'cell1_002_004.npy',
                    'cell1_002_005.npy',
                    'cell1_002_006.npy',
                    ],
                'e': [
                    'data_20120417_cell1_002_e.npy',
                ],
                'f': [
                    'data_20120417_cell1_002_f.npy',
                ],
                's': [
                    'data_20120417_cell1_002_s.npy',
                ],
            },
            {
                'id': 3,
                'cell_id': 1,
                'name': '20120417_cell3',
                'images': [
                    'cell3_002_001.npy',
                    'cell3_002_002.npy',
                    'cell3_002_003.npy',
                    'cell3_002_004.npy',
                    'cell3_002_005.npy',
                    'cell3_002_006.npy',
                    ],
                'e': [
                    'data_20120417_cell3_002_e.npy',
                ],
                'f': [
                    'data_20120417_cell3_002_f.npy',
                ],
                's': [
                    'data_20120417_cell3_002_s.npy',
                ],
            },
            {
                'id': 4,
                'cell_id': 1,
                'name': '20120417_cell3',
                'images': [
                    'cell3_003_001.npy',
                    'cell3_003_002.npy',
                    'cell3_003_003.npy',
                    'cell3_003_004.npy',
                    'cell3_003_005.npy',
                    'cell3_003_006.npy',
                    ],
                'e': [
                    'data_20120417_cell3_003_e.npy',
                ],
                'f': [
                    'data_20120417_cell3_003_f.npy',
                ],
                's': [
                    'data_20120417_cell3_003_s.npy',
                ],
            },
            {
                'id': 5,
                'cell_id': 2,
                'name': '20120417_cell4',
                'images': [
                    'cell4_001_001.npy',
                    'cell4_001_002.npy',
                    'cell4_001_003.npy',
                    'cell4_001_004.npy',
                    'cell4_001_005.npy',
                    'cell4_001_006.npy',
                    ],
                'e': [
                    'data_20120417_cell4_001_e.npy',
                ],
                'f': [
                    'data_20120417_cell4_001_f.npy',
                ],
                's': [
                    'data_20120417_cell4_001_s.npy',
                ],
            },
            {
                'id': 6,
                'cell_id': 2,
                'name': '20120417_cell4',
                'images': [
                    'cell4_002_001.npy',
                    'cell4_002_002.npy',
                    'cell4_002_003.npy',
                    'cell4_002_004.npy',
                    'cell4_002_005.npy',
                    'cell4_002_006.npy',
                    ],
                'e': [
                    'data_20120417_cell4_002_e.npy',
                ],
                'f': [
                    'data_20120417_cell4_002_f.npy',
                ],
                's': [
                    'data_20120417_cell4_002_s.npy',
                ],
            },
            {
                'id': 7,
                'cell_id': 2,
                'name': '20120417_cell4',
                'images': [
                    'cell4_003_001.npy',
                    'cell4_003_002.npy',
                    'cell4_003_003.npy',
                    'cell4_003_004.npy',
                    'cell4_003_005.npy',
                    'cell4_003_006.npy',
                    ],
                'e': [
                    'data_20120417_cell4_003_e.npy',
                ],
                'f': [
                    'data_20120417_cell4_003_f.npy',
                ],
                's': [
                    'data_20120417_cell4_003_s.npy',
                ],
            },
        ]

        if not os.path.exists(self.numpy) or self.overwrite_numpy:
            images, ss = {}, {}
            spikes, time_e = {}, {}
            corr_f, raw_f, time_f, df_f, cell_ids = {}, {}, {}, {}, {}
            selected_spikes = {}
            for d in cell_data:
                images[d['id']] = []
                spikes[d['id']] = []
                time_e[d['id']] = []
                corr_f[d['id']] = []
                raw_f[d['id']] = []
                time_f[d['id']] = []
                ss[d['id']] = []
                df_f[d['id']] = []
                cell_ids[d['id']] = []

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
                if self.use_df_f:
                    for idx in range(self.df_f_window, len(corr_f[d['id']])):
                        event_start = idx - self.df_f_window
                        df_mean = corr_f[d['id']][event_start:idx]
                        df_f[d['id']] += [(
                            corr_f[d['id']][idx] - df_mean) / df_mean]
                    df_f[d['id']] = np.concatenate(df_f[d['id']]).squeeze()

                # Repeat cell ids
                cell_ids[d['id']] = (
                    np.ones(
                        (len(corr_f[d['id']]))) * d['cell_id']).astype(
                            np.int64)

                # Here's where it gets funky
                time_f[d['id']] = np.asarray([
                    float(
                        format(x, '.4f')) for x in time_f[d['id']]]).squeeze()
                selected_spikes[d['id']] = []
                form_e = [float(format(f, '.4f')) for f in time_e[d['id']]]
                for idx in tqdm(
                        range(1, len(time_f[d['id']])), desc='Binning spikes'):
                    it_start = float(format(time_f[d['id']][idx - 1], '.4f'))
                    it_end = float(format(time_f[d['id']][idx], '.4f'))
                    e_start = form_e.index(it_start)
                    e_end = form_e.index(it_end)
                    binned_spikes = np.sum(
                        [spikes[d['id']][x] for x in range(e_start, e_end)])
                    selected_spikes[d['id']] += [binned_spikes]
                it_end = float(format(time_f[d['id']][0], '.4f'))
                e_end = form_e.index(it_end)
                first_event = np.sum(
                    [spikes[d['id']][x] for x in range(0, e_end)])
                selected_spikes[d['id']] = np.concatenate(
                    ([first_event],
                        np.asarray(selected_spikes[d['id']]).squeeze()))

                # If self.use_df_f, trim spikes
                if self.use_df_f:
                    selected_spikes[d['id']] = selected_spikes[d['id']][
                        self.df_f_window:]

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

            # Add the cell_id
            cat_labels = np.concatenate([v for v in selected_spikes.values()])
            cat_ids = np.concatenate([v for v in cell_ids.values()])

            num_images = len(cat_images)
            num_labels = len(cat_labels)
            assert num_images == num_labels, 'Different numbers of ims/labs'

            # Save all processed data
            np.savez(
                self.numpy,
                cat_images=cat_images,
                cat_labels=cat_labels,
                cat_ids=cat_ids)
        else:
            cached_data = np.load(self.numpy)
            print 'Loading data'
            cat_images = cached_data['cat_images']
            cat_labels = cached_data['cat_labels']
            cat_ids = cached_data['cat_ids']
            num_images = len(cat_images)
            num_labels = len(cat_labels)
            assert num_images == num_labels, 'Different numbers of ims/labs'

        # Cut into distinct events -- start with reshape for this
        num_events = int(num_images / self.timepoints)
        total_events = num_events * self.timepoints
        cat_images = cat_images[:total_events]
        full_cat_labels = cat_labels[:total_events]
        full_cat_ids = cat_ids[:total_events]
        im_shape = cat_images.shape
        cat_images = cat_images.reshape(
            num_events,
            self.timepoints,
            im_shape[1],
            im_shape[2],
            im_shape[3]).astype(np.float32)
        cat_labels = full_cat_labels.reshape(num_events, -1).astype(np.int64)
        cat_ids = full_cat_ids.reshape(num_events, -1).astype(np.int64)

        # Split into train/test
        cat_labels = np.expand_dims(cat_labels.sum(-1), axis=-1)
        cat_ids = np.expand_dims(cat_ids[:, 0], axis=-1)

        # Combine cat_labels and cat_ids
        cat_labels = np.concatenate((cat_labels, cat_ids), axis=-1)
        if self.binarize_spikes:
            cat_labels[cat_labels[:, 0] > 1, 0] = 1

        # Cross validate with a proportion of images from each session
        train_images, train_labels, train_ids = [], [], []
        test_images, test_labels, test_ids = [], [], []
        for cid in np.unique(cat_ids):
            cidx = (cat_ids == cid).squeeze()
            cell_ims = cat_images[cidx]
            cell_labels = cat_labels[cidx]
            cell_ids = cat_ids[cidx]
            cv_split = np.round(cidx.sum() * self.train_prop).astype(int)
            train_images += [cell_ims[:cv_split]]
            train_labels += [cell_labels[:cv_split]]
            train_ids += [cell_ids[:cv_split]]
            test_images += [cell_ims[cv_split:]]
            test_labels += [cell_labels[cv_split:]]
            test_ids += [cell_ids[cv_split:]]

        train_images = np.concatenate(train_images)
        train_labels = np.concatenate(train_labels)
        train_ids = np.concatenate(train_ids)
        test_images = np.concatenate(test_images)
        test_labels = np.concatenate(test_labels)
        test_ids = np.concatenate(test_ids)

        # cv_split = np.round(num_events * self.train_prop).astype(int)
        # full_cv_split = np.round(
        #     len(full_cat_ids) * self.train_prop).astype(int)
        # train_images = cat_images[:cv_split]
        # test_images = cat_images[cv_split:]
        # train_labels = cat_labels[:cv_split]
        # test_labels = cat_labels[cv_split:]

        # Fix imbalance with repetitions if requested
        if self.fix_imbalance_train:
            # Train repeat
            spike_idx = train_labels[:, 0] > 0
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
                (train_labels, rep_spike_labels), axis=0)

        if self.fix_imbalance_test:
            spike_idx = test_labels[:, 0] > 0
            num_spikes = spike_idx.sum()
            num_events = len(test_labels)
            num_neg = num_events - num_spikes
            rep = int(num_neg / num_spikes)
            spike_images = test_images[spike_idx.squeeze()]
            rep_spike_images = spike_images.repeat(rep, axis=0)
            rep_spike_labels = test_labels[spike_idx].repeat(rep, axis=0)
            test_images = np.concatenate(
                (test_images, rep_spike_images), axis=0)
            test_labels = np.concatenate(
                (test_labels, rep_spike_labels), axis=0)

        if self.shuffle_train:
            rand_order = np.random.permutation(len(train_images))
            train_images = train_images[rand_order]
            train_labels = train_labels[rand_order]

        if self.shuffle_test:
            rand_order = np.random.permutation(len(test_images))
            test_images = test_images[rand_order]
            test_labels = test_labels[rand_order]

        if self.validation_slice > 0:
            # Reshape test_images and test_labels
            im_dims = test_images.shape
            slice_test_images = test_images.reshape(
                [np.prod(im_dims[:2]).tolist()] + list(im_dims[2:]))
            slice_test_labels = test_labels[:, 0].repeat(
                self.timepoints, axis=-1).reshape(-1, 1)
            slice_test_ids = test_labels[:, 1].repeat(
                self.timepoints, axis=-1).reshape(-1, 1)
            # slice_test_labels = full_cat_labels[full_cv_split:]
            # slice_test_ids = full_cat_ids[full_cv_split:]

            # Stagger into events separated by self.validation_slice
            num_staggered_events = len(
                slice_test_images)
            num_zeros = num_staggered_events / self.validation_slice
            staggered_test_images = np.zeros((
                num_zeros,
                self.timepoints,
                im_dims[2],
                im_dims[3],
                im_dims[4]))
            staggered_test_labels = np.zeros((
                num_zeros, 1))
            staggered_test_ids = np.zeros((
                num_zeros, 1))
            start = 0
            for idx in range(num_staggered_events):
                end = start + self.timepoints
                if end > num_staggered_events:
                    break
                else:
                    staggered_test_images[idx] = slice_test_images[
                        start:end]
                    staggered_test_labels[idx] = slice_test_labels[
                        start:end].sum()
                    staggered_test_ids[idx] = slice_test_ids[start]
                start += self.validation_slice

            df_check = staggered_test_images.sum(
                1, keepdims=True).sum(
                    2, keepdims=True).sum(
                        3, keepdims=True).sum(
                            4, keepdims=True).squeeze()
            dfs = df_check != 0.
            test_images = staggered_test_images[dfs].astype(np.float32)
            test_labels = np.concatenate((
                staggered_test_labels[dfs],
                staggered_test_ids[dfs]), axis=-1).astype(np.int64)

        # Sum labels per event (total spikes)
        files = {
            'train': train_images,
            'test': test_images
        }
        labels = {
            'train': train_labels.tolist(),
            'test': test_labels.tolist()
        }
        print 'Spikes in training: %s' % np.sum(train_labels[:, 0])
        print 'Spikes in testing: %s' % np.sum(test_labels[:, 0])
        print 'Unique groups: %s' % np.unique(train_labels[:, 1])
        return files, labels

    def get_data(self):
        files, labels = self.pull_data()
        return files, labels
