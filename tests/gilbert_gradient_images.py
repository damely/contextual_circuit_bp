import os
import json
import time
import numpy as np
import tensorflow as tf
import experiments
from db import db
from config import Config
from argparse import ArgumentParser
from utils import logger
from utils import py_utils
from ops import data_loader
from ops import model_utils
from ops import loss_utils
from ops import eval_metrics
from ops import hp_opt_utils
from ops import tf_fun
from tensorboard.plugins.pr_curve import summary as pr_summary
from tensorflow.python.framework import ops


@ops.RegisterGradient('GradLRP')
def _GradLRP(op, grad):
    eps = 1e-12
    op_out = op.outputs[0]
    op_in = op.inputs[0]
    return grad * op_out / (op_in + eps)


def print_model_architecture(model_summary):
    """Print a list fancy."""
    print '_' * 20
    print 'Model architecture:'
    print '_' * 20
    for s in model_summary:
        print s
    print '_' * 20


def add_to_config(d, config):
    """Add attributes to config class."""
    for k, v in d.iteritems():
        if isinstance(v, list) and len(v) == 1:
            v = v[0]
        setattr(config, k, v)
    return config


def process_DB_exps(experiment_name, log, config):
    """Interpret and prepare hyperparams at runtime."""
    exp_params, exp_id = db.get_parameters(
        experiment_name=experiment_name,
        log=log,
        evaluation=config.load_and_evaluate_ckpt)
    if exp_id is None:
        err = 'No empty experiments found.' + \
            'Did you select the correct experiment name?'
        log.fatal(err)
        raise RuntimeError(err)
    for k, v in exp_params.iteritems():
        if isinstance(v, basestring) and '{' in v and '}' in v:
            v = v.strip('{').strip('}').split(',')
        setattr(config, k, v)
    if not hasattr(config, '_id'):
        config._id = exp_id
    return config, exp_params


def get_data_pointers(dataset, base_dir, cv, log):
    """Get data file pointers."""
    data_pointer = os.path.join(base_dir, '%s_%s.tfrecords' % (dataset, cv))
    data_means = os.path.join(base_dir, '%s_%s_means.npy' % (dataset, cv))
    log.info(
        'Using %s tfrecords: %s' % (
            cv,
            data_pointer)
        )
    py_utils.check_path(
        data_pointer, log, '%s not found.' % data_pointer)
    mean_loc = py_utils.check_path(
        data_means, log, '%s not found for cv: %s.' % (data_means, cv))
    data_means_image, data_means_label = None, None
    if not mean_loc:
        alt_data_pointer = data_means.replace('.npy', '.npz')
        alt_data_pointer = py_utils.check_path(
            alt_data_pointer, log, '%s not found.' % alt_data_pointer)
        # TODO: Fix this API and make it more flexible. Kill npzs in Allen?
        if not alt_data_pointer:
            # No mean for this dataset
            data_means = None
        else:
            log.info('Loading means from npz for cv: %s.' % cv)
            data_means = np.load(alt_data_pointer)
            data_means_vol = data_means[data_means.keys()[0]].item()
            if 'image' in data_means_vol.keys():
                data_means_image = data_means_vol['image']
            if 'label' in data_means_vol.keys():
                data_means_label = data_means_vol['label']
    else:
        data_means_image = np.load(data_means)
    return data_pointer, data_means_image, data_means_label


def main(
        experiment_name,
        list_experiments=False,
        load_and_evaluate_ckpt=None,
        config_file=None,
        ckpt_file=None,
        gpu_device='/gpu:0'):
    """Create a tensorflow worker to run experiments in your DB."""
    if list_experiments:
        exps = db.list_experiments()
        print '_' * 30
        print 'Initialized experiments:'
        print '_' * 30
        for l in exps:
            print l.values()[0]
        print '_' * 30
        if len(exps) == 0:
            print 'No experiments found.'
        else:
            print 'You can add to the DB with: '\
                'python prepare_experiments.py --experiment=%s' % \
                exps[0].values()[0]
        return

    if experiment_name is None:
        print 'No experiment specified. Pulling one out of the DB.'
        experiment_name = db.get_experiment_name()

    # Prepare to run the model
    config = Config()
    condition_label = '%s_%s' % (experiment_name, py_utils.get_dt_stamp())
    experiment_label = '%s' % (experiment_name)
    log = logger.get(os.path.join(config.log_dir, condition_label))
    experiment_dict = experiments.experiments()[experiment_name]()
    config = add_to_config(d=experiment_dict, config=config)  # Globals
    config.load_and_evaluate_ckpt = load_and_evaluate_ckpt
    config, exp_params = process_DB_exps(
        experiment_name=experiment_name,
        log=log,
        config=config)  # Update config w/ DB params
    config = np.load(config_file).item()
    dataset_module = py_utils.import_module(
        model_dir=config.dataset_info,
        dataset=config.dataset)
    dataset_module = dataset_module.data_processing()  # hardcoded class name
    train_data, train_means_image, train_means_label = get_data_pointers(
        dataset=config.dataset,
        base_dir=config.tf_records,
        cv=dataset_module.folds.keys()[1],  # TODO: SEARCH FOR INDEX.
        log=log
    )
    val_data, val_means_image, val_means_label = get_data_pointers(
        dataset=config.dataset,
        base_dir=config.tf_records,
        cv=dataset_module.folds.keys()[0],
        log=log
    )

    # Initialize output folders
    dir_list = {
        'checkpoints': os.path.join(
            config.checkpoints, condition_label),
        'summaries': os.path.join(
            config.summaries, condition_label),
        'condition_evaluations': os.path.join(
            config.condition_evaluations, condition_label),
        'experiment_evaluations': os.path.join(  # DEPRECIATED
            config.experiment_evaluations, experiment_label),
        'visualization': os.path.join(
            config.visualizations, condition_label),
        'weights': os.path.join(
            config.condition_evaluations, condition_label, 'weights')
    }
    [py_utils.make_dir(v) for v in dir_list.values()]

    # Prepare data loaders on the cpu
    if all(isinstance(i, list) for i in config.data_augmentations):
        if config.data_augmentations:
            config.data_augmentations = py_utils.flatten_list(
                config.data_augmentations,
                log)
    config.epochs = 1
    config.shuffle = False
    with tf.device('/cpu:0'):
        train_images, train_labels = data_loader.inputs(
            dataset=train_data,
            batch_size=config.batch_size,
            model_input_image_size=dataset_module.model_input_image_size,
            tf_dict=dataset_module.tf_dict,
            data_augmentations=config.data_augmentations,
            num_epochs=config.epochs,
            tf_reader_settings=dataset_module.tf_reader,
            shuffle=config.shuffle_train,
            resize_output=config.resize_output)
        if hasattr(config, 'val_augmentations'):
            val_augmentations = config.val_augmentations
        else:
            val_augmentations = config.data_augmentations
        val_images, val_labels = data_loader.inputs(
            dataset=val_data,
            batch_size=config.batch_size,
            model_input_image_size=dataset_module.model_input_image_size,
            tf_dict=dataset_module.tf_dict,
            data_augmentations=['resize_and_crop'],
            num_epochs=config.epochs,
            tf_reader_settings=dataset_module.tf_reader,
            shuffle=config.shuffle_val,
            resize_output=config.resize_output)
    log.info('Created tfrecord dataloader tensors.')

    # Load model specification
    struct_name = config.model_struct.split(os.path.sep)[-1]
    try:
        model_dict = py_utils.import_module(
            dataset=struct_name,
            model_dir=os.path.join(
                'models',
                'structs',
                experiment_name).replace(os.path.sep, '.')
            )
    except IOError:
        print 'Could not find the model structure: %s in folder %s' % (
            struct_name,
            experiment_name)

    # Inject model_dict with hyperparameters if requested
    model_dict.layer_structure = hp_opt_utils.inject_model_with_hps(
        layer_structure=model_dict.layer_structure,
        exp_params=exp_params)

    # Prepare model on GPU
    with tf.device(gpu_device):
        with tf.variable_scope('cnn') as scope:
            # Normalize labels if needed
            if 'normalize_labels' in exp_params.keys():
                if exp_params['normalize_labels'] == 'zscore':
                    train_labels -= train_means_label['mean']
                    train_labels /= train_means_label['std']
                    log.info('Z-scoring labels.')
                elif exp_params['normalize_labels'] == 'mean':
                    train_labels -= train_means_label['mean']
                    log.info('Mean-centering labels.')

            # Training model
            if len(dataset_module.output_size) == 2:
                log.warning(
                    'Found > 1 dimension for your output size.'
                    'Converting to a scalar.')
                dataset_module.output_size = np.prod(
                    dataset_module.output_size)

            if hasattr(model_dict, 'output_structure'):
                # Use specified output layer
                output_structure = model_dict.output_structure
            else:
                output_structure = None
            model = model_utils.model_class(
                mean=train_means_image,
                training=True,
                output_size=dataset_module.output_size)
            train_scores, model_summary = model.build(
                data=train_images,
                layer_structure=model_dict.layer_structure,
                output_structure=output_structure,
                log=log,
                tower_name='cnn')
            eval_graph = tf.Graph()
            with eval_graph.as_default():
                with eval_graph.gradient_override_map(
                        {'selu': 'GradLRP'}):
                    train_grad_images = tf.gradients(
                        train_scores[0] * tf.cast(train_labels, tf.float32),
                        train_images)[0]
            log.info('Built training model.')
            log.debug(
                json.dumps(model_summary, indent=4),
                verbose=0)
            print_model_architecture(model_summary)

            # Check the shapes of labels and scores
            if not isinstance(train_scores, list):
                if len(
                        train_scores.get_shape()) != len(
                            train_labels.get_shape()):
                    train_shape = train_scores.get_shape().as_list()
                    label_shape = train_labels.get_shape().as_list()
                    if len(
                        train_shape) == 2 and len(
                            label_shape) == 1 and train_shape[-1] == 1:
                        train_labels = tf.expand_dims(train_labels, axis=-1)
                    elif len(
                        train_shape) == 2 and len(
                            label_shape) == 1 and train_shape[-1] == 1:
                        train_scores = tf.expand_dims(train_scores, axis=-1)

            # Prepare the loss function
            train_loss, _ = loss_utils.loss_interpreter(
                logits=train_scores,  # TODO
                labels=train_labels,
                loss_type=config.loss_function,
                weights=config.loss_weights,
                dataset_module=dataset_module)

            # Add loss tensorboard tracking
            if isinstance(train_loss, list):
                for lidx, tl in enumerate(train_loss):
                    tf.summary.scalar('training_loss_%s' % lidx, tl)
                train_loss = tf.add_n(train_loss)
            else:
                tf.summary.scalar('training_loss', train_loss)

            # Add weight decay if requested
            if len(model.regularizations) > 0:
                train_loss = loss_utils.wd_loss(
                    regularizations=model.regularizations,
                    loss=train_loss,
                    wd_penalty=config.regularization_strength)
            train_op = loss_utils.optimizer_interpreter(
                loss=train_loss,
                lr=config.lr,
                optimizer=config.optimizer,
                constraints=config.optimizer_constraints,
                model=model)
            log.info('Built training loss function.')

            # Add a score for the training set
            train_accuracy = eval_metrics.metric_interpreter(
                metric=dataset_module.score_metric,  # TODO: Attach to exp cnfg
                pred=train_scores,  # TODO
                labels=train_labels)

            # Add aux scores if requested
            train_aux = {}
            if hasattr(dataset_module, 'aux_scores'):
                for m in dataset_module.aux_scores:
                    train_aux[m] = eval_metrics.metric_interpreter(
                        metric=m,
                        pred=train_scores,
                        labels=train_labels)[0]  # TODO: Fix for multiloss

            # Prepare remaining tensorboard summaries
            if len(train_images.get_shape()) == 4:
                tf_fun.image_summaries(train_images, tag='Training images')
            if len(train_labels.get_shape()) > 2:
                tf_fun.image_summaries(
                    train_labels,
                    tag='Training_targets')
                tf_fun.image_summaries(
                    train_scores,
                    tag='Training_predictions')
            if isinstance(train_accuracy, list):
                for tidx, ta in enumerate(train_accuracy):
                    tf.summary.scalar('training_accuracy_%s' % tidx, ta)
            else:
                tf.summary.scalar('training_accuracy', train_accuracy)
            if config.pr_curve:
                if isinstance(train_scores, list):
                    for pidx, train_score in enumerate(train_scores):
                        train_label = train_labels[:, pidx]
                        pr_summary.op(
                            tag='training_pr_%s' % pidx,
                            predictions=tf.cast(
                                tf.argmax(
                                    train_score,
                                    axis=-1),
                                tf.float32),
                            labels=tf.cast(train_label, tf.bool),
                            display_name='training_precision_recall_%s' % pidx)
                else:
                    pr_summary.op(
                        tag='training_pr',
                        predictions=tf.cast(
                            tf.argmax(
                                train_scores,
                                axis=-1),
                            tf.float32),
                        labels=tf.cast(train_labels, tf.bool),
                        display_name='training_precision_recall')
            log.info('Added training summaries.')

            # Validation model
            scope.reuse_variables()
            val_model = model_utils.model_class(
                mean=train_means_image,  # Normalize with train data
                training=False,  # False,
                output_size=dataset_module.output_size)
            val_scores, _ = val_model.build(  # Ignore summary
                data=val_images,
                layer_structure=model_dict.layer_structure,
                output_structure=output_structure,
                log=log,
                tower_name='cnn')
            eval_graph = tf.Graph()
            with eval_graph.as_default():
                with eval_graph.gradient_override_map(
                        {'selu': 'GradLRP'}):
                    val_grad_images = tf.gradients(
                        val_scores[0] * tf.cast(val_labels, tf.float32),
                        val_images)[0]
            log.info('Built validation model.')

            # Check the shapes of labels and scores
            if not isinstance(train_scores, list):
                if len(val_scores.get_shape()) != len(val_labels.get_shape()):
                    val_shape = val_scores.get_shape().as_list()
                    val_label_shape = val_labels.get_shape().as_list()
                    if len(
                        val_shape) == 2 and len(
                            val_label_shape) == 1 and val_shape[-1] == 1:
                        val_labels = tf.expand_dims(val_labels, axis=-1)
                    if len(
                        val_shape) == 2 and len(
                            val_label_shape) == 1 and val_shape[-1] == 1:
                        val_scores = tf.expand_dims(val_scores, axis=-1)
            val_loss, _ = loss_utils.loss_interpreter(
                logits=val_scores,
                labels=val_labels,
                loss_type=config.loss_function,
                weights=config.loss_weights,
                dataset_module=dataset_module)

            # Add loss tensorboard tracking
            if isinstance(val_loss, list):
                for lidx, tl in enumerate(val_loss):
                    tf.summary.scalar('validation_loss_%s' % lidx, tl)
                val_loss = tf.add_n(val_loss)
            else:
                tf.summary.scalar('validation_loss', val_loss)

            # Add a score for the validation set
            val_accuracy = eval_metrics.metric_interpreter(
                metric=dataset_module.score_metric,  # TODO
                pred=val_scores,
                labels=val_labels)

            # Add aux scores if requested
            val_aux = {}
            if hasattr(dataset_module, 'aux_scores'):
                for m in dataset_module.aux_scores:
                    val_aux[m] = eval_metrics.metric_interpreter(
                        metric=m,
                        pred=val_scores,
                        labels=val_labels)[0]  # TODO: Fix for multiloss

            # Prepare tensorboard summaries
            if len(val_images.get_shape()) == 4:
                tf_fun.image_summaries(val_images, tag='Validation')
            if len(val_labels.get_shape()) > 2:
                tf_fun.image_summaries(
                    val_labels,
                    tag='Validation_targets')
                tf_fun.image_summaries(
                    val_scores,
                    tag='Validation_predictions')
            if isinstance(val_accuracy, list):
                for vidx, va in enumerate(val_accuracy):
                    tf.summary.scalar('validation_accuracy_%s' % vidx, va)
            else:
                tf.summary.scalar('validation_accuracy', val_accuracy)
            if config.pr_curve:
                if isinstance(val_scores, list):
                    for pidx, val_score in enumerate(val_scores):
                        val_label = val_labels[:, pidx]
                        pr_summary.op(
                            tag='validation_pr_%s' % pidx,
                            predictions=tf.cast(
                                tf.argmax(
                                    val_score,
                                    axis=-1),
                                tf.float32),
                            labels=tf.cast(val_label, tf.bool),
                            display_name='validation_precision_recall_%s' %
                            pidx)
                else:
                    pr_summary.op(
                        tag='validation_pr',
                        predictions=tf.cast(
                            tf.argmax(
                                val_scores,
                                axis=-1),
                            tf.float32),
                        labels=tf.cast(val_labels, tf.bool),
                        display_name='validation_precision_recall')
            log.info('Added validation summaries.')

    # Set up summaries and saver
    saver = tf.train.Saver(tf.global_variables())

    # Initialize the graph
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

    # Need to initialize both of these if supplying num_epochs to inputs
    sess.run(
        tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer())
        )

    # Set up exemplar threading
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # Create dictionaries of important training and validation information
    train_dict = {
        'train_loss': train_loss,
        'train_images': train_images,
        'train_labels': train_labels,
        'train_op': train_op,
        'train_scores': train_scores,
        'train_grad_images': train_grad_images
    }
    val_dict = {
        'val_loss': val_loss,
        'val_images': val_images,
        'val_labels': val_labels,
        'val_scores': val_scores,
        'val_grad_images': val_grad_images
    }
    if isinstance(train_accuracy, list):
        for tidx, (ta, va) in enumerate(zip(train_accuracy, val_accuracy)):
            train_dict['train_accuracy_%s' % tidx] = ta
            val_dict['val_accuracy_%s' % tidx] = va
    else:
        train_dict['train_accuracy_0'] = train_accuracy
        val_dict['val_accuracy_0'] = val_accuracy

    if load_and_evaluate_ckpt is not None:
        # Remove the train operation and add a ckpt pointer
        del train_dict['train_op']

    if hasattr(dataset_module, 'aux_score'):
        # Attach auxillary scores to tensor dicts
        for m in dataset_module.aux_scores:
            train_dict['train_aux_%s' % m] = train_aux[m]
            val_dict['val_aux_%s' % m] = val_aux[m]

    # Start training loop
    checkpoint_dir = dir_list['checkpoints']
    step = 0
    train_losses, train_accs, train_aux, timesteps = {}, {}, {}, {}
    val_scores, val_aux, val_labels, val_grads = {}, {}, {}, {}
    train_images, val_images = {}, {}
    train_scores, train_labels = {}, {}
    train_aux_check = np.any(['aux_score' in k for k in train_dict.keys()])
    val_aux_check = np.any(['aux_score' in k for k in val_dict.keys()])

    # Restore model
    saver.restore(sess, ckpt_file)

    # Start evaluation
    try:
        while not coord.should_stop():
            start_time = time.time()
            train_vars = sess.run(train_dict.values())
            it_train_dict = {k: v for k, v in zip(
                train_dict.keys(), train_vars)}
            duration = time.time() - start_time
            train_losses[step] = it_train_dict['train_loss']
            train_accs[step] = it_train_dict['train_accuracy_0']
            train_images[step] = it_train_dict['train_images']
            train_labels[step] = it_train_dict['train_labels']
            train_scores[step] = it_train_dict['train_scores']
            timesteps[step] = duration
            if train_aux_check:
                # Loop through to find aux scores
                it_train_aux = {
                    itk: itv
                    for itk, itv in it_train_dict.iteritems()
                    if 'aux_score' in itk}
                train_aux[step] = it_train_aux
            assert not np.isnan(
                it_train_dict['train_loss']
                ).any(), 'Model diverged with loss = NaN'
            if step % config.validation_iters == 0:
                it_val_scores, it_val_labels, it_val_aux, it_val_grads, it_val_ims = [], [], [], [], []
                for num_vals in range(config.num_validation_evals):
                    # Validation accuracy as the average of n batches
                    val_vars = sess.run(val_dict.values())
                    it_val_dict = {k: v for k, v in zip(
                        val_dict.keys(), val_vars)}
                    it_val_labels += [it_val_dict['val_labels']]
                    it_val_scores += [it_val_dict['val_scores']]
                    it_val_grads += [it_val_dict['val_grad_images']]
                    it_val_ims += [it_val_dict['val_images']]
                    if val_aux_check:
                        iva = {
                            itk: itv
                            for itk, itv in it_val_dict.iteritems()
                            if 'aux_score' in itk}
                        it_val_aux += [iva]
                val_scores[step] = it_val_scores
                val_labels[step] = it_val_labels
                val_aux[step] = it_val_aux
                val_images[step] = it_val_grads
                val_grads[step] = it_val_ims

            # End iteration
            step += 1

    except tf.errors.OutOfRangeError:
        print 'Done with evaluation for %d epochs, %d steps.' % (
            config.epochs,
            step)
        print 'Saved to: %s' % checkpoint_dir
    finally:
        coord.request_stop()
    coord.join(threads)
    sess.close()

    import ipdb;ipdb.set_trace()
    np.savez(
        'val_imgs_grads',
        val_images=val_images,  # it_val_dict['val_images'],
        val_grads=val_grads,  # it_val_dict['val_grad_images'],
        val_labels=val_labels,  # it_val_dict['val_labels'],
        val_scores=val_scores)  # it_val_dict['val_scores'][0])


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--experiment',
        dest='experiment_name',
        type=str,
        default=None,
        help='Name of the experiment.')
    parser.add_argument(
        '--config',
        dest='config_file',
        type=str,
        default='/media/data_cifs/contextual_circuit/condition_evaluations/snakes_256_2018_03_16_22_03_29/training_config_file.npy',
        help='Location of config file.')
    parser.add_argument(
        '--ckpt',
        dest='ckpt_file',
        type=str,
        default='/media/data_cifs/contextual_circuit/checkpoints/snakes_256_2018_03_16_22_03_29/model_32000.ckpt-32000',
        help='Location of config file.')
    parser.add_argument(
        '--list_experiments',
        dest='list_experiments',
        action='store_true',
        help='Name of the experiment.')
    # TODO: Add the ability to specify multiple GPUs for parallelization.
    args = parser.parse_args()
    main(**vars(args))
