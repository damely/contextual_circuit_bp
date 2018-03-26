import os
import numpy as np
from db import db
from db import credentials
import experiments
from config import Config
from argparse import ArgumentParser
import pandas as pd
from utils import py_utils
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import plotly.plotly as py
import plotly.tools as tls


def plot_with_plotly(plotly_fig, chart):
    try:
        plot_url = py.plot(plotly_fig, auto_open=False)
        print 'Uploaded %s chart to: %s' % (chart, plot_url)
    except:
        print 'Failed to upload to plotly.'


def main(
        experiment_name,
        im_ext='.pdf',
        transform_loss=None,  # 'log',
        colors='Paired',
        flip_axis=False,
        exclude=None):
    """Plot results of provided experiment name."""
    config = Config()
    pl_creds = credentials.plotly_credentials()
    py.sign_in(
        pl_creds['username'],
        pl_creds['api_key'])

    # Get experiment data
    perf = db.get_performance(experiment_name=experiment_name)
    if len(perf) == 0:
        raise RuntimeError('Could not find any results.')
    structure_names = [x['model_struct'].split('/')[-1] for x in perf]
    optimizers = [x['optimizer'] for x in perf]
    lrs = [x['lr'] for x in perf]
    datasets = [x['dataset'] for x in perf]
    loss_funs = [x['loss_function'] for x in perf]
    optimizers = [x['optimizer'] for x in perf]
    wd_types = [x['regularization_type'] for x in perf]
    wd_penalties = [x['regularization_strength'] for x in perf]
    steps = [float(x['training_step']) for x in perf]
    training_loss = [float(x['training_loss']) for x in perf]
    validation_loss = [float(x['validation_loss']) for x in perf]
    timesteps = [0. if x['timesteps'] is None else float(x['timesteps']) for x in perf]
    ckpts = [x['ckpt_file'] for x in perf]
    summaries = [x['summary_dir'] for x in perf]
    u_t = [0. if x['u_t'] is None else float(x['u_t']) for x in perf]
    q_t = [0. if x['q_t'] is None else float(x['q_t']) for x in perf]
    p_t = [0. if x['p_t'] is None else float(x['p_t']) for x in perf]
    t_t = [0. if x['t_t'] is None else float(x['t_t']) for x in perf]

    # Pass data into a pandas DF
    model_params = [
        '%s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s' % (
            ipa,
            ipb,
            ipc,
            ipd,
            ipe,
            ipf,
            ipg,
            iph,
            ipi,
            ipj,
            ipk,
            ipl,
            ipm)
        for ipa, ipb, ipc, ipd, ipe, ipf, ipg, iph, ipi, ipj, ipk, ipl, ipm
        in zip(
            structure_names,
            optimizers,
            lrs,
            loss_funs,
            optimizers,
            wd_types,
            wd_penalties,
            datasets,
            timesteps,
            u_t,
            q_t,
            p_t,
            t_t)]
    model_infos = ['%s | %s' % (ckpt, summ) for ckpt, summ in zip(ckpts, summaries)]

    # DF and plot
    df = pd.DataFrame(
        np.vstack(
            (
                model_params,
                model_infos,
                steps,
                training_loss,
                validation_loss
            )
        ).transpose(),
        columns=[
            'model parameters',
            'model_infos',
            'training iteration',
            'training loss',
            'validation loss'
            ]
        )
    df['training iteration'] = pd.to_numeric(
        df['training iteration'],
        errors='coerce')
    df['training loss'] = pd.to_numeric(df['training loss'], errors='coerce')

    if exclude is not None:
        exclusion_search = df['model parameters'].str.contains(exclude)
        df = df[exclusion_search == False]
        print 'Removed %s rows.' % exclusion_search.sum()

    # Start plotting
    experiment_dict = experiments.experiments()[experiment_name]()
    print 'Plotting results for dataset: %s.' % experiment_dict['dataset'][0]
    dataset_module = py_utils.import_module(
        model_dir=config.dataset_info,
        dataset=experiment_dict['dataset'][0])
    dataset_module = dataset_module.data_processing()  # hardcoded class name
    if transform_loss is None:
        loss_label = ''
    elif transform_loss == 'log':
        loss_label = ' log loss'
        df['training loss'] = np.log(df['training loss'])
    elif transform_loss == 'max':
        loss_label = ' normalized (x / max(x)) '
        df['training loss'] /= df.groupby(
            'model parameters')['training loss'].transform(max)
    if ['loss_function'] in experiment_dict.keys():
        loss_metric = experiment_dict['loss_function'][0]
    else:
        loss_metric = dataset_module.default_loss_function
    df['validation loss'] = pd.to_numeric(df['validation loss'])
    if loss_metric == 'pearson':
        loss_label = 'Pearson correlation' + loss_label
    elif loss_metric == 'l2':
        loss_label = 'L2' + loss_label
    else:
        loss_label = 'Classification accuracy (%)'
        df['validation loss'] *= 100.

    if ['score_metric'] in experiment_dict.keys():
        score_metric = experiment_dict['score_metric']
    else:
        score_metric = dataset_module.score_metric
    if score_metric == 'pearson':
        y_lab = 'Pearson correlation'

    matplotlib.style.use('ggplot')
    plt.rc('font', size=6)
    plt.rc('legend', fontsize=8, labelspacing=3)
    f, axs = plt.subplots(2, figsize=(20, 30))
    ax = axs[1]
    NUM_COLORS = len(df['model parameters'].unique())
    cm = plt.get_cmap('gist_rainbow')
    ax.set_color_cycle([cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])
    for k in df['model parameters'].unique():
        tmp = df[df['model parameters'] == k]
        tmp = tmp.sort_values('training iteration', ascending=False)
        ax = tmp.plot(
            x='training iteration',
            y='training loss',
            label=k,
            kind='line',
            ax=ax,
            logy=False)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title('Training')
    ax.set_ylabel(loss_label)
    # ax.legend_.remove()
    ax = axs[0]
    ax.set_color_cycle([cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])
    for k in df['model parameters'].unique():
        tmp = df[df['model parameters'] == k]
        tmp = tmp.sort_values('training iteration', ascending=False)
        ax = tmp.plot(
            x='training iteration',
            y='validation loss',
            label=k,
            kind='line',
            ax=ax,
            logy=False)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title('Validation')
    # TODO: Mine the experiment declarations for the appropos metric name.
    # ax.set_ylabel(y_lab)
    # ax.legend_.remove()
    out_name = os.path.join(
        config.plots,
        '%s_%s%s' % (
            experiment_name, py_utils.get_dt_stamp(), im_ext))
    plt.savefig(out_name)
    print 'Saved to: %s' % out_name
    plotly_fig = tls.mpl_to_plotly(f)
    plotly_fig['layout']['autosize'] = True
    # plotly_fig['layout']['showlegend'] = True
    plot_with_plotly(plotly_fig, 'line')
    plt.close(f)

    # Plot max performance bar graph
    f = plt.figure()
    max_perf = df.groupby(
        ['model parameters'], as_index=False)['validation loss'].max()
    plt.rc('xtick', labelsize=2)
    ax = max_perf.plot.bar(
        x='model parameters', y='validation loss', legend=False)
    plt.tight_layout()
    ax.set_title('Max validation value')
    # ax.set_ylabel(y_lab)
    out_name = os.path.join(
        config.plots,
        '%s_%s_bar%s' % (
            experiment_name, py_utils.get_dt_stamp(), im_ext))
    plt.savefig(out_name)
    print 'Saved to: %s' % out_name
    try:
        plotly_fig = tls.mpl_to_plotly(f)
        plot_with_plotly(plotly_fig, chart='bar')
    except Exception as e:
        print 'Failed to plot bar chart in plotly: %s' % e
    plt.close(f)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--experiment',
        dest='experiment_name',
        type=str,
        default=None,
        help='Name of the experiment.')
    parser.add_argument(
        '--exclude',
        dest='exclude',
        type=str,
        default=None,
        help='Experiment exclusion keyword.')
    parser.add_argument(
        '--flip_axis',
        dest='flip_axis',
        type=str,
        default=None,
        help='Flip x axis.')
    args = parser.parse_args()
    main(**vars(args))
