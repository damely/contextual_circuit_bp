import os
import numpy as np
from glob import glob
from sklearn import metrics
from matplotlib import pyplot as plt
from tqdm import tqdm


data_dir = '/media/data_cifs/contextual_circuit/condition_evaluations/'
suffix_dir = 'weights'
models = {
    'crcns_2d_2018_01_26_16_39_59': ['crcns_2d', '2d narrow GRU'],
    'crcns_2d_2018_01_26_16_40_01': ['crcns_2d', '2d wide GRU'],
    'crcns_1d_2018_01_26_16_40_08': ['crcns_1d', '1d GRU'],
}
output_file = 'f1_scores.png'  # output_file = 'accuracy_scores.png'
ylim = [0, 1]
optimize = [0, 'f1_score']  # optimize = [0, 'accuracy_score']
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
            'classification_report'
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
            # 'confusion_matrix'
            'classification_report'
        ],
    }
}

results = {}
eval_traces = {}
for d, model in tqdm(models.iteritems(), total=len(models)):
    files = glob(
        os.path.join(
            data_dir,
            d,
            suffix_dir,
            '*.npy'))
    model = model[0]
    trim_files = [int(f.split(os.path.sep)[-1].split('_')[2]) for f in files]
    evals = np.unique(trim_files)
    metric_dict = {}
    top_f1s = []
    for it in evals:
        it_files = glob(
            os.path.join(
                data_dir,
                d,
                suffix_dir,
                '%s_%s_*.npy' % (model, it)))
        score_files = [f for f in it_files if 'val_' in f]
        assert len(score_files) == 2, 'Error in globbing'
        label_file = [f for f in score_files if 'labels' in f][0]
        score_file = [f for f in score_files if 'scores' in f][0]
        labels = np.load(label_file).item()
        scores = np.load(score_file).item()
        metric_output = {}
        for k, v in targets.iteritems():
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
            for m in v['metrics']:
                metric = getattr(metrics, m)
                metric_output[k][m] = metric(
                    y_true=it_labels,
                    y_pred=it_scores)
        metric_dict[it] = metric_output
    optim_scores = []
    for it in evals:
        optim_scores += [metric_dict[it][optimize[0]][optimize[1]]]
    optim_scores = np.asarray(optim_scores)
    top_eval = evals[np.argmax(optim_scores)]
    eval_traces[d] = optim_scores
    results[d] = metric_dict[top_eval]

# Plot results
f = plt.figure()
for idx, (k, v) in enumerate(eval_traces.iteritems()):
    plt.plot(v, label=models[k][1])
plt.xlabel('Epoch of training')
plt.ylabel(optimize[1].replace('_', ' '))
plt.ylim(ylim)
plt.legend()
plt.savefig(output_file)
plt.show()
plt.close(f)
