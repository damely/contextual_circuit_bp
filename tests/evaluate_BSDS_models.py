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
image_data = data_files['files'].item()['train']
label_data = data_files['labels'].item()['train']

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
    'contours_2018_02_28_17_33_35',
    'model_19000.ckpt-19000')

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
labs, scores = main.main(
    experiment_name=exp_name,
    load_and_evaluate_ckpt=sel_ckpts,
    placeholder_data=placeholder_data)




preds = np.argmax(all_scores, axis=-1)
bin_labs = (np.copy(all_labs) > 0).astype(int)
acc = np.mean(bin_labs == preds)
f_score = metrics.f1_score(y_true=bin_labs, y_pred=preds)
p, r, thresh = metrics.precision_recall_curve(
    bin_labs,
    all_scores[:, 1],
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
