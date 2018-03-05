import os
import time
import tensorflow as tf
import numpy as np
import prepare_experiments
from utils import py_utils
from datetime import datetime
from ops import hp_opt_utils


def check_early_stop(
        perf_history,
        minimum_length=20,
        short_history=3,
        long_history=5,
        fail_function=np.less_equal):
    """
    Determine whether to stop early. Using deepgaze criteria:

    We determine this point by comparing the performance from
    the last three epochs to the performance five epochs before those.
    Training runs for at least 20 epochs, and is terminated if all three
    of the last epochs show decreased performance or if
    800 epochs are reached.

    """
    if len(perf_history) < minimum_length:
        early_stop = False
    else:
        short_perf = perf_history[-short_history:]
        long_perf = perf_history[-long_history + short_history:short_history]
        short_check = fail_function(np.mean(long_perf), short_perf)
        if all(short_check):  # If we should stop
            early_stop = True
        else:
            early_stop = False

    return early_stop


def training_loop(
        config,
        db,
        coord,
        sess,
        summary_op,
        summary_writer,
        saver,
        threads,
        summary_dir,
        checkpoint_dir,
        weight_dir,
        train_dict,
        val_dict,
        train_model,
        val_model,
        exp_params,
        performance_metric='validation_loss',
        aggregator='max'):
    """Run the model training loop."""
    step, time_elapsed = 0, 0
    train_losses, train_accs, train_aux, timesteps = {}, {}, {}, {}
    val_losses, val_accs, val_scores, val_aux, val_labels = {}, {}, {}, {}, {}
    train_aux_check = np.any(['aux_score' in k for k in train_dict.keys()])
    val_aux_check = np.any(['aux_score' in k for k in val_dict.keys()])
    if config.save_weights:
        weight_dict = {
            k[0]: v for k, v in val_model.var_dict.iteritems() if k[1] == 0}
        val_dict = dict(
            val_dict,
            **weight_dict)
        # weight_dict = {
        #     k[0]: v
        #       for k, v in train_model.var_dict.iteritems() if k[1] == 0}
        # train_dict = dict(
        #     train_dict,
        #     **weight_dict)
    try:
        while not coord.should_stop():
            start_time = time.time()
            train_vars = sess.run(train_dict.values())
            it_train_dict = {k: v for k, v in zip(
                train_dict.keys(), train_vars)}
            duration = time.time() - start_time
            train_losses[step] = it_train_dict['train_loss']
            train_accs[step] = it_train_dict['train_accuracy_0']
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
                it_val_acc = np.asarray([])
                it_val_loss = np.asarray([])
                it_val_scores, it_val_labels, it_val_aux = [], [], []
                for num_vals in range(config.num_validation_evals):
                    # Validation accuracy as the average of n batches
                    val_vars = sess.run(val_dict.values())
                    it_val_dict = {k: v for k, v in zip(
                        val_dict.keys(), val_vars)}
                    it_val_acc = np.append(
                        it_val_acc,
                        it_val_dict['val_accuracy_0'])
                    it_val_loss = np.append(
                        it_val_loss,
                        it_val_dict['val_loss'])
                    it_val_labels += [it_val_dict['val_labels']]
                    it_val_scores += [it_val_dict['val_scores']]
                    if val_aux_check:
                        iva = {
                            itk: itv
                            for itk, itv in it_val_dict.iteritems()
                            if 'aux_score' in itk}
                        it_val_aux += [iva]
                val_acc = it_val_acc.mean()
                val_lo = it_val_loss.mean()
                val_accs[step] = val_acc
                val_losses[step] = val_lo
                val_scores[step] = it_val_scores
                val_labels[step] = it_val_labels
                val_aux[step] = it_val_aux

                # Summaries
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)

                # Training status and validation accuracy
                format_str = (
                    '%s: step %d, loss = %.2f (%.1f examples/sec; '
                    '%.3f sec/batch) | Training accuracy = %s | '
                    'Validation accuracy = %s | logdir = %s')
                print format_str % (
                    datetime.now(),
                    step,
                    it_train_dict['train_loss'],
                    config.batch_size / duration,
                    float(duration),
                    it_train_dict['train_accuracy_0'],
                    val_acc,
                    summary_dir)

                # Save the model checkpoint if it's the best yet
                if config.top_n_validation > 0:
                    rep_idx = val_acc > val_accs
                    if sum(rep_idx) > 0:
                        force_save = True
                        val_accs[np.argmax(rep_idx)] = val_acc
                else:
                    force_save = True

                if force_save:
                    ckpt_path = os.path.join(
                        checkpoint_dir,
                        'model_' + str(step) + '.ckpt')
                    saver.save(
                        sess,
                        ckpt_path,
                        global_step=step)
                    print 'Saved checkpoint to: %s' % ckpt_path
                    force_save = False
                    time_elapsed += float(duration)
                    db.update_performance(
                        experiment_id=config._id,
                        experiment_name=config.experiment_name,
                        summary_dir=summary_dir,
                        ckpt_file=ckpt_path,
                        training_loss=float(it_train_dict['train_loss']),
                        validation_loss=float(val_acc),
                        time_elapsed=time_elapsed,
                        training_step=step)
                    if config.save_weights:
                        it_weights = {
                            k: it_val_dict[k] for k in weight_dict.keys()}
                        py_utils.save_npys(
                            data=it_weights,
                            model_name='%s_%s' % (
                                config.experiment_name,
                                step),
                            output_string=weight_dir)
                    if config.save_validation_predictions:
                        if isinstance(val_scores[step][0], list):
                            save_val_scores, save_val_labels = {}, {}
                            for vidx in range(len(val_scores[step][0])):
                                save_val_scores[vidx] = []
                                save_val_labels[vidx] = []
                            for vs, vl in zip(
                                    val_scores[step],
                                    val_labels[step]):
                                for vidx in range(len(vs)):
                                    save_val_scores[vidx] += [vs[vidx]]
                                    save_val_labels[vidx] += [vl[:, vidx]]
                            for vidx in range(len(val_scores[step][0])):
                                save_val_scores[vidx] = np.concatenate(
                                    save_val_scores[vidx])
                                save_val_labels[vidx] = np.concatenate(
                                    save_val_labels[vidx])
                        else:
                            save_val_scores = val_scores[step]
                            save_val_labels = val_labels[step]

                        save_val_data = {
                            'val_scores': save_val_scores,
                            'val_labels': save_val_labels
                        }
                        py_utils.save_npys(
                            data=save_val_data,
                            model_name='%s_%s' % (
                                config.experiment_name,
                                step),
                            output_string=weight_dir)
                if config.early_stop:
                    keys = np.sort([int(k) for k in val_accs.keys()])
                    sorted_vals = np.asarray([val_accs[k] for k in keys])
                    if check_early_stop(sorted_vals):
                        print 'Triggered an early stop.'
                        break
            else:
                # Training status
                format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; '
                              '%.3f sec/batch) | Training accuracy = %s')
                print format_str % (
                    datetime.now(),
                    step,
                    it_train_dict['train_loss'],
                    config.batch_size / duration,
                    float(duration),
                    it_train_dict['train_accuracy_0'])

            # End iteration
            step += 1

    except tf.errors.OutOfRangeError:
        print 'Done training for %d epochs, %d steps.' % (config.epochs, step)
        print 'Saved to: %s' % checkpoint_dir
    finally:
        coord.request_stop()
    coord.join(threads)
    sess.close()

    # If using online hp optimization, update the database with performance
    if exp_params['hp_current_iteration'] is not None:

        # If we have not exceeded the maximum online hp optimizations:
        if exp_params['hp_current_iteration'] < exp_params['hp_max_studies']:

            # Database lookup to get all performance for this hp-thread
            performance_history = db.query_hp_hist(
                exp_params=exp_params,
                performance_metric=performance_metric,
                aggregator=aggregator)

            # Call on online optimization tools
            exp_params = hp_opt_utils.hp_optim_interpreter(
                performance_history=performance_history,
                aggregator=aggregator)

            # Prepare parameters for DB
            pk = prepare_experiments.protected_keys()
            exp_params = prepare_experiments.prepare_hp_params(
                parameter_dict=exp_params,
                pk=pk)

            # Iterate the count
            exp_params['hp_current_iteration'] += 1
            for k, v in exp_params.iteritems():
                if isinstance(v, basestring) and 'null' in v:
                    exp_params[k] = None

            # Update the database with the new hyperparameters
            db.update_online_experiment(
                exp_combos=[exp_params],
                experiment_link=exp_params['experiment_link'])

    # Package output variables into a dictionary
    output_dict = {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'train_aux': train_aux,
        'timesteps': timesteps,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'val_scores': val_scores,
        'val_labels': val_labels,
        'val_aux': val_aux,
    }
    return output_dict
