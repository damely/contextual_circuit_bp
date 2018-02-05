import os
import numpy as np
from glob import glob
from sklearn import metrics
from matplotlib import pyplot as plt
from tqdm import tqdm
import pandas as pd
import seaborn as sns
sns.set()


def deconv_model_info(pull_extra=True):
    data_dir = '/media/data_cifs/contextual_circuit/condition_evaluations/'
    suffix_dir = 'weights'
    models = {
        # 'crcns_2d_2018_01_26_16_39_59': ['crcns_2d', '2d narrow GRU', 'r'],
        # 'crcns_2d_2018_01_26_16_40_01': ['crcns_2d', '2d wide GRU', 'g'],

        # 'crcns_2d_2018_01_26_19_15_07': ['crcns_2d', '2d narrow GRU 2', 'r'],
        # 'crcns_2d_2018_01_27_14_36_28': ['crcns_2d', '2d wide GRU', 'b'],
        # 'crcns_2d_2018_01_27_14_36_23': ['crcns_2d', '2d wide GRU', 'b'],
        # 'crcns_2d_2018_01_27_10_18_17': ['crcns_2d', '2d wide GRU', 'b'],
        # 'crcns_1d_2018_01_26_16_40_08': ['crcns_1d', '1d GRU_1', 'gray'],
        # 'crcns_1d_2018_01_26_19_00_53': ['crcns_1d', '1d GRU_2', 'gray'],
        # 'crcns_1d_2018_01_27_10_01_48': ['crcns_1d', 'best 1d', 'g'],
        # 'crcns_1d_2018_01_26_19_00_53': ['crcns_1d', 'worst 1d', 'g'],
        # 'crcns_1d_2018_01_27_09_19_54': ['crcns_1d', 'best 1d 2', 'g'],
        # 'crcns_1d_2018_01_27_10_01_48': ['crcns_1d', 'worst 1d 2', 'g'],

        # 'crcns_2d_2018_01_27_11_20_19': ['crcns_2d', 'best 2d', 'k'],
        # 'crcns_2d_2018_01_27_10_29_21': ['crcns_2d', 'worst 2d', 'k'],
        # 'crcns_2d_2018_01_27_11_20_20': ['crcns_2d', 'best 2d 2', 'k'],
        # 'crcns_2d_2018_01_27_10_18_17': ['crcns_2d', 'worst 2d 2', 'k'],
    }
    if os.path.exists(os.path.join('tests', 'summaries.csv')) and pull_extra:
        models_1d = pd.read_csv(
            os.path.join('tests', 'summaries.csv')).as_matrix()
        models_1d = [
            m[0].split(os.path.sep)[-1]
            for m in models_1d if os.path.exists(m[0])]
        for idx, m in enumerate(models_1d):
            mname = '_'.join(m.split(os.path.sep)[-1].split('_')[:4])
            if '1d' in mname:
                color = 'gray'
            else:
                color = 'b'
            models[m] = [mname, '%s GRU %s' % (m.split('_')[0], idx), color]
    return models, suffix_dir, data_dir


models, suffix_dir, data_dir = deconv_model_info()
output_file = 'f1_scores.png'  # output_file = 'accuracy_scores.png'
ylim = [0, 1]
optimize = [0, 'f1_score']
# optimize = [0, 'accuracy_score']
# optimize = [0, 'normalized_mutual_info_score']
targets = {
    0: {
        'label': 'spikes',
        'binarize_labels': True,
        'binarize_scores': False,
        'argmax_labels': False,
        'argmax_scores': True,
        'metrics': [
            'accuracy_score',
            'f1_score',
            # 'precision_recall_curve',
            # 'confusion_matrix'
            # 'classification_report',
            # 'confusion_matrix',
            # 'normalized_mutual_info_score'
        ],
        },
    1: {
        'label': 'cells',
        'binarize_labels': False,
        'binarize_scores': False,
        'argmax_labels': False,
        'argmax_scores': True,
        'metrics': [
            # 'accuracy_score',
            # 'f1_score',
            # 'precision_recall_curve',
            'confusion_matrix',
            # 'classification_report'
        ],
    }
}

results = {}
eval_traces = {}
selected_scores = {}
selected_labels = {}
top_eval_data = {}
top_eval_ckpts = {}
for d, model in tqdm(models.iteritems(), total=len(models)):
    files = glob(
        os.path.join(
            data_dir,
            d,
            suffix_dir,
            '*.npy'))
    model = model[0]
    trim_files = [int(f.split(os.path.sep)[-1].split('_')[4]) for f in files]
    evals = np.unique(trim_files)
    metric_dict = {}
    all_scores = {}
    all_labels = {}
    for it in evals:
        it_files = glob(
            os.path.join(
                data_dir,
                d,
                suffix_dir,
                '%s_%s_*.npy' % (model, it)))

        score_files = [f for f in it_files if 'val_' in f]
        assert len(score_files) == 2, 'Error in globbing %s' % d
        label_file = [f for f in score_files if 'labels' in f][0]
        score_file = [f for f in score_files if 'scores' in f][0]
        try:
            labels = np.load(label_file).item()
            scores = np.load(score_file).item()
        except:
            labels = np.load(label_file)
            scores = np.load(score_file)
            labels = {0: labels.reshape(-1, 1)}
            scores = {0: scores.reshape(-1, 2)}
        metric_output = {}
        eval_scores = {}
        eval_labels = {}
        for k, v in targets.iteritems():
            if k in labels.keys():
                metric_output[k] = {}
                it_labels = np.copy(labels[k])
                it_scores = np.copy(scores[k])
                if v['binarize_labels']:
                    it_labels[it_labels > 1] = 1
                if v['binarize_scores']:
                    it_scores[it_scores > 1] = 1
                if v['argmax_labels']:
                    if len(it_labels.shape) == 0:
                        it_labels = np.expand_dims(it_labels, axis=-1)
                    it_labels = np.argmax(it_labels, axis=-1)
                if v['argmax_scores']:
                    if len(it_scores.shape) == 0:
                        it_scores = np.expand_dims(it_scores, axis=-1)
                    it_scores = np.argmax(it_scores, axis=-1)
                # eval_scores[k] = it_scores
                # eval_labels[k] = it_labels
                for m in v['metrics']:
                    metric = getattr(metrics, m)
                    metric_output[k][m] = metric(
                        it_labels,
                        it_scores)
        metric_dict[it] = metric_output
        # all_scores[it] = eval_scores
        # all_labels[it] = eval_labels
    optim_scores = []
    for it in evals:
        optim_scores += [metric_dict[it][optimize[0]][optimize[1]]]
    optim_scores = np.asarray(optim_scores)
    top_eval = evals[np.argmax(optim_scores)]
    eval_traces[d] = optim_scores
    results[d] = metric_dict[top_eval]
    sel_data_path = os.path.sep.join(files[0].split(os.path.sep)[:-1])
    sel_ckpt_path = os.path.sep.join(files[0].split(os.path.sep)[:-4])
    top_eval_data[d] = os.path.join(
        sel_data_path,
        '%s_%s' % (model, top_eval))
    top_eval_ckpts[d] = os.path.join(
        sel_ckpt_path,
        'checkpoints',
        d,
       'model_%s.ckpt-%s' % (top_eval, top_eval))
    # selected_scores[d] = all_scores[top_eval]
    # selected_labels[d] = all_labels[top_eval]

# Plot results
f = plt.figure()
labs = []
scs = []
for idx, (k, v) in enumerate(eval_traces.iteritems()):
    plt.plot(v, label=models[k][1], c=models[k][2])
    labs += [k]
    scs += [v]
plt.xlabel('Epoch of training')
plt.ylabel(optimize[1].replace('_', ' '))
plt.ylim(ylim)
plt.legend()
plt.savefig(output_file)
plt.show()
plt.close(f)

# Save ckpts for the top 1d and 2d models
model_names = np.asarray(
    labs)[np.argsort(np.asarray([np.max(sc) for sc in scs]))]
models_1d = [m for m in model_names if '1d' in m]
models_2d = [m for m in model_names if '2d' in m]
np.savez(
    os.path.join('tests', 'top_models'),
    models_1d=models_1d,
    models_2d=models_2d)

# # Plot dissimilarity matrices
# num_models = len(models)
# model_names = [models[k][1] for k in selected_scores.keys()]
# ss = np.concatenate(
#     [v[0][None, :] for v in selected_scores.values()],
#         axis=0)
# sl = np.concatenate(
#     [v[0][None, :] for v in selected_labels.values()],
#         axis=0)
# df = pd.DataFrame(ss.transpose(), columns=model_names)
# 
# # Create a categorical palette to identify the networks
# network_pal = sns.husl_palette(len(model_names), s=.45)
# network_lut = dict(zip(map(str, model_names), network_pal))
# 
# # Convert the palette to vectors that will be drawn on the side of the matrix
# network_colors = pd.Series(model_names, index=df.columns).map(network_lut)
# 
# # Draw the full plot
# sns.clustermap(
#     df.corr(),
#     center=0,
#     cmap="RdBu",
#     row_colors=network_colors,
#     col_colors=network_colors,
#     linewidths=.75,
#     figsize=(13, 13))
# plt.show()

