import time
import tensorflow as tf
import numpy as np
from utils import py_utils


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


def evaluation_loop(
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
        placeholder_data=None,
        performance_metric='validation_loss',
        aggregator='max'):
    """Run the model training loop."""
    step = 0
    train_losses, train_accs, train_aux, timesteps = {}, {}, {}, {}
    val_scores, val_aux, val_labels = {}, {}, {}
    train_images, val_images = {}, {}
    train_scores, train_labels = {}, {}
    train_aux_check = np.any(['aux_score' in k for k in train_dict.keys()])
    val_aux_check = np.any(['aux_score' in k for k in val_dict.keys()])

    # Restore model
    saver.restore(sess, config.load_and_evaluate_ckpt)

    # Start evaluation
    if config.save_weights:
        weight_dict = {
            k[0]: v for k, v in train_model.var_dict.iteritems() if k[1] == 0}
        val_dict = dict(
            val_dict,
            **weight_dict)
    if placeholder_data is not None:
        num_batches = len(placeholder_data['label_data']) // config.batch_size
        batch_index = np.arange(num_batches).repeat(config.batch_size)
        for idx in np.arange(num_batches):
            batch_images = placeholder_data['image_data'][batch_index == idx]
            batch_labels = placeholder_data['label_data'][batch_index == idx]
            batch_images = batch_images.reshape(
                placeholder_data['val_image_shape'])
            batch_labels = batch_labels.reshape(
                placeholder_data['val_label_shape'])
            feed_dict = {
                placeholder_data['train_images']: batch_images,
                placeholder_data['train_labels']: batch_labels,
            }
            it_vars = sess.run(train_dict.values(), feed_dict=feed_dict)
            it_dict = {k: v for k, v in zip(
                train_dict.keys(), it_vars)}
            import ipdb;ipdb.set_trace()
            a = 2

    else:
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
                    it_val_scores, it_val_labels, it_val_aux = [], [], []
                    for num_vals in range(config.num_validation_evals):
                        # Validation accuracy as the average of n batches
                        val_vars = sess.run(val_dict.values())
                        it_val_dict = {k: v for k, v in zip(
                            val_dict.keys(), val_vars)}
                        it_val_labels += [it_val_dict['val_labels']]
                        it_val_scores += [it_val_dict['val_scores']]
                        if val_aux_check:
                            iva = {
                                itk: itv
                                for itk, itv in it_val_dict.iteritems()
                                if 'aux_score' in itk}
                            it_val_aux += [iva]
                    val_scores[step] = it_val_scores
                    val_labels[step] = it_val_labels
                    val_aux[step] = it_val_aux
                    val_images[step] = it_val_dict['val_images']

                    # Save the model checkpoint if it's the best yet
                    it_weights = {
                        k: it_val_dict[k] for k in weight_dict.keys()}
                    py_utils.save_npys(
                        data=it_weights,
                        model_name='%s_%s' % (
                            config.experiment_name,
                            step),
                        output_string=weight_dir)
                import ipdb;ipdb.set_trace()

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

    # Package images into a dictionary
    image_dict = {
        'train_images': train_images,
        'val_images': val_images,
        'train_scores': train_scores,
        'train_labels': train_labels,
        'val_scores': val_scores,
        'val_labels': val_labels
    }
    py_utils.save_npys(
        data=image_dict,
        model_name='%s_%s' % (
            config.experiment_name,
            step),
        output_string=weight_dir)

    return val_labels, val_scores

    # Package output variables into a dictionary
    # output_dict = {
    #     'train_losses': train_losses,
    #     'train_accs': train_accs,
    #     'train_aux': train_aux,
    #     'timesteps': timesteps,
    #     'val_losses': val_losses,
    #     'val_accs': val_accs,
    #     'val_scores': val_scores,
    #     'val_labels': val_labels,
    #     'val_aux': val_aux,
    # }
    # return output_dict
