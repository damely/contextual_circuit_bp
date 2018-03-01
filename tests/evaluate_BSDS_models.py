"""Evaluate a model for the Bush-Vannevar grant."""
import os
import main
import numpy as np
import tensorflow as tf
from glob import glob
from sklearn import metrics
from matplotlib import pyplot as plt
from dataset_processing import BSDS500
from scipy import misc
from matplotlib import gridspec


def sigmoid(x):
    """Apply a sigmoid nonlinearity."""
    return 1/(1+np.exp(-x))


def plot_mosaic(
        maps,
        title='Mosaic',
        rc=None,
        cc=None,
        show_plot=True):
    if rc is None:
        rc = np.ceil(np.sqrt(len(maps))).astype(int)
        cc = np.ceil(np.sqrt(len(maps))).astype(int)
    f = plt.figure(figsize=(10, 10))
    plt.suptitle(title, fontsize=20)
    gs1 = gridspec.GridSpec(rc, cc)
    gs1.update(wspace=0.01, hspace=0.01)  # set the spacing between axes.
    for idx, im in enumerate(maps):
        ax1 = plt.subplot(gs1[idx])
        plt.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.set_aspect('equal')
        ax1.imshow(im.squeeze())
    if show_plot:
        plt.show()
        plt.close(f)


def get_ckpt(path):
    ckpt_pointers = glob(os.path.join(
        path,
        '*ckpt*'))
    ckpt_pointers = [x for x in ckpt_pointers if 'meta' in x]
    ckpt_pointer = sorted(
        ckpt_pointers,
        key=lambda file: os.path.getctime(file))[-1].split('.meta')[0]
    return ckpt_pointer


exp_name = 'contours'
batch_size = 10
use_placeholders = False

# Load validation data
data_file = BSDS500.data_processing().get_files()
data_files = np.load(
    os.path.join(
        '%smedia' % os.path.sep,
        'data_cifs',
        'image_datasets',
        'BSDS500',
        'images',
        'val',
        'file_paths.npz'))
image_data = data_files['files'].item()['val']
label_data = data_files['labels'].item()['val']

# Get model
ckpt_dir = os.path.join(
    '%smedia' % os.path.sep,
    'data_cifs',
    'contextual_circuit',
    'checkpoints')
sel_ckpts = os.path.join(
    '%smedia' % os.path.sep,
    'data_cifs',
    'contextual_circuit',
    'checkpoints',
    'contours_2018_03_01_16_14_02',  # 1l cnn: contours_2018_02_28_14_17_57
    'model_1000.ckpt-1000')

# Get image/label info
images = np.asarray([misc.imread(im) for im in image_data]).astype(np.float32)
labels = np.asarray([np.load(lab) for lab in label_data]).astype(np.float32)
example_image = images[0]
example_label = labels[0]
placeholder_data = {
    'train_image_shape': [batch_size] + [x for x in example_image.shape],
    'train_image_dtype': tf.float32,
    'train_label_shape': [batch_size] + [x for x in example_label.shape] + [1],
    'train_label_dtype': tf.float32,
    'val_image_shape': [batch_size] + [x for x in example_image.shape],
    'val_image_dtype': tf.float32,
    'val_label_shape': [batch_size] + [x for x in example_label.shape] + [1],
    'val_label_dtype': tf.float32,
    'image_data': images,
    'label_data': labels
}
if use_placeholders:
    labs, scores = main.main(
        experiment_name=exp_name,
        load_and_evaluate_ckpt=sel_ckpts,
        placeholder_data=placeholder_data)
else:
    labs, scores = main.main(
        experiment_name=exp_name,
        load_and_evaluate_ckpt=sel_ckpts)

if len(scores.keys()) > 1:
    raise RuntimeError
scores = scores[0]
labs = labs[0]

# Create a mosaic
plot_mosaic(images.astype(np.uint8), title='Images', rc=10, cc=10)
plot_mosaic(labs, rc=10, cc=10, title='Labels', show_plot=True)
# plot_mosaic(labels, rc=10, cc=10, title='Labels', show_plot=True)
plot_mosaic(scores, rc=10, cc=10, title='Predictions', show_plot=True)

# Evaluate performance 
map_score = metrics.average_precision_score(
    labs.reshape(batch_size, -1),
    scores.reshape(batch_size, -1))
p, r, thresh = metrics.precision_recall_curve(
    labs.reshape(batch_size, -1),
    scores.reshape(batch_size, -1),
    pos_label=1)
plt.step(
    r,
    p,
    color='b',
    alpha=0.2,
    where='post')
plt.fill_between(
    r,
    p,
    step='post',
    alpha=0.2,
    color='b')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.show()
# np.savez(
#     version,
#     preds=preds,
#     all_scores=all_scores,
#     bin_labs=bin_labs,
#     acc=acc,
#     f_score=f_score,
#     p=p,
#     r=r,
#     model=sel_ckpts,
#     thresh=thresh)
