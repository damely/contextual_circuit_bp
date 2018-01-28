"""Evaluate a model for the Bush-Vannevar grant."""
import os
import main
import numpy as np
from glob import glob


def get_ckpt(path):
    ckpt_pointers = glob(os.path.join(
        path,
        '*ckpt*'))
    ckpt_pointers = [x for x in ckpt_pointers if 'meta' in x]
    ckpt_pointer = sorted(
        ckpt_pointers,
        key=lambda file: os.path.getctime(file))[-1].split('.meta')[0]
    return ckpt_pointer


ckpt_dir = os.path.join(
    '%smedia' % os.path.sep,
    'data_cifs',
    'contextual_circuit',
    'checkpoints')
model_files = np.load(os.path.join('tests', 'top_models.npz'))
model_1d = model_files['models_1d'][0]
model_2d = model_files['models_2d'][0]
model_1d_ckpt = get_ckpt(os.path.join(ckpt_dir, model_1d))
model_2d_ckpt = get_ckpt(os.path.join(ckpt_dir, model_2d))
labs, scores = main.main(
    experiment_name='crcns_2d',
    load_and_evaluate_ckpt=model_2d_ckpt)
all_scores = np.concatenate([sc[0] for sc in scores[0]])
all_labs = np.concatenate(labs[0])

