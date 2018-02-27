"""Routines for encoding data into TFrecords."""
import numpy as np
import tensorflow as tf
from scipy import misc
from tqdm import tqdm
from utils import image_processing


def load_image(f, im_size, reshape=True):
    """Load image and convert it to a 4D tensor."""
    image = misc.imread(f).astype(np.float32)
    if len(image.shape) < 3 and reshape:  # Force H/W/C
        image = np.repeat(image[:, :, None], im_size[-1], axis=-1)
    return image


def create_example(data_dict):
    """Create entry in tfrecords."""
    return tf.train.Example(
        # Example contains a Features proto object
        features=tf.train.Features(
            # Features has a map of string to Feature proto objects
            feature=data_dict
        )
    )


def preprocess_image(image, preprocess, im_size):
    """Preprocess image files before encoding in TFrecords."""
    if 'crop_center' in preprocess:
        image = image_processing.crop_center(image, im_size)
    elif 'resize' in preprocess:
        image = image_processing.resize(image, im_size)
    elif 'pad_resize' in preprocess:
        image = image_processing.pad_square(image)
        image = image_processing.resize(image, im_size)
    return image


def encode_tf(encoder, x):
    """Process data for TFRecords."""
    encoder_name = encoder.func_name
    if 'bytes' in encoder_name:
        return encoder(x.tostring())
    else:
        return encoder(x)


def data_to_tfrecords(
        files,
        labels,
        targets,
        ds_name,
        im_size,
        label_size,
        preprocess,
        store_z=False):
    """Convert dataset to tfrecords."""
    print 'Building dataset: %s' % ds_name
    for idx, ((fk, fv), (lk, lv)) in enumerate(
        zip(
            files.iteritems(),
            labels.iteritems())):
        it_ds_name = '%s_%s.tfrecords' % (ds_name, fk)
        if store_z:
            means = []
        else:
            means = np.zeros((im_size))
        with tf.python_io.TFRecordWriter(it_ds_name) as tfrecord_writer:
            image_count = 0
            for it_f, it_l in tqdm(
                    zip(fv, lv), total=len(fv), desc='Building %s' % fk):
                if isinstance(it_f, basestring):
                    if '.npy' in it_f:
                        image = np.load(it_f)
                    else:
                        image = load_image(it_f, im_size).astype(np.float32)
                        image = preprocess_image(image, preprocess, im_size)
                else:
                    image = it_f
                if store_z:
                    means += [image]
                else:
                    means += image
                if isinstance(it_l, basestring):
                    if '.npy' in it_l:
                        label = np.load(it_l)
                    else:
                        label = load_image(
                            it_l, label_size, reshape=False).astype(np.float32)
                        label = preprocess_image(it_l, preprocess, label_size)
                else:
                    label = it_l
                data_dict = {
                    'image': encode_tf(targets['image'], image),
                    'label': encode_tf(targets['label'], label)
                }
                example = create_example(data_dict)
                if example is not None:
                    # Keep track of how many images we use
                    image_count += 1
                    # use the proto object to serialize the example to a string
                    serialized = example.SerializeToString()
                    # write the serialized object to disk
                    tfrecord_writer.write(serialized)
                    example = None
            if store_z:
                means = np.asarray(means).reshape(len(means), -1)
                np.savez(
                    '%s_%s_means' % (ds_name, fk),
                    image={
                        'mean': means.mean(),
                        'std': means.std()
                        })
            else:
                np.save(
                    '%s_%s_means' % (ds_name, fk), means / float(image_count))
            print 'Finished %s with %s images (dropped %s)' % (
                it_ds_name, image_count, len(fv) - image_count)
