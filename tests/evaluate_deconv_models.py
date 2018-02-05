"""Evaluate a model for the Bush-Vannevar grant."""
import os
import main
import numpy as np
from glob import glob
from sklearn import metrics
from matplotlib import pyplot as plt


def get_ckpt(path):
    ckpt_pointers = glob(os.path.join(
        path,
        '*ckpt*'))
    ckpt_pointers = [x for x in ckpt_pointers if 'meta' in x]
    ckpt_pointer = sorted(
        ckpt_pointers,
        key=lambda file: os.path.getctime(file))[-1].split('.meta')[0]
    return ckpt_pointer


version = '2d'
ckpt_dir = os.path.join(
    '%smedia' % os.path.sep,
    'data_cifs',
    'contextual_circuit',
    'checkpoints')
model_files = np.load(os.path.join('tests', 'top_models.npz'))
if version == '1d':
    model_1d = model_files['models_1d'][-1]
    model_1d_ckpt = get_ckpt(os.path.join(ckpt_dir, model_1d))
    sel_ckpts = model_1d_ckpt
    sel_ckpts = '/media/data_cifs/contextual_circuit/checkpoints/crcns_1d_two_loss_2018_01_27_23_50_39/model_49000.ckpt-49000'  # 2loss
elif version == '2d':
    model_2d = model_files['models_2d'][-1]
    model_2d_ckpt = get_ckpt(os.path.join(ckpt_dir, model_2d))
    sel_ckpts = model_2d_ckpt
    # sel_ckpts = '/media/data_cifs/contextual_circuit/checkpoints/crcns_2d_two_loss_2018_01_27_22_14_08/model_9000.ckpt-9000'  # 2loss
    # sel_ckpts = '/media/data_cifs/contextual_circuit/checkpoints/crcns_2d_two_loss_2018_01_28_17_03_53/model_7000.ckpt-7000'  # 1loss
else:
    raise NotImplementedError
print '*' * 60
print sel_ckpts
print '*' * 60
exp_name = '_'.join(sel_ckpts.split(os.path.sep)[-2].split('_')[:4])
labs, scores = main.main(
    experiment_name=exp_name,
    load_and_evaluate_ckpt=sel_ckpts)

if len(scores[0][0]) == 2:
    all_scores = np.concatenate([sc[0] for sc in scores[0]])
    all_labs = np.concatenate(labs[0])
    all_labs = all_labs[:, 0]
else:
    all_scores = np.concatenate([sc for sc in scores[0]])
    all_labs = np.concatenate(labs[0])
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
np.savez(
    version,
    preds=preds,
    all_scores=all_scores,
    bin_labs=bin_labs,
    acc=acc,
    f_score=f_score,
    p=p,
    r=r,
    model=sel_ckpts,
    thresh=thresh)
