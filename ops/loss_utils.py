import numpy as np
import tensorflow as tf
from ops.eval_metrics import pearson_score


def optimizer_interpreter(
        loss,
        lr,
        optimizer,
        model=None,
        constraints=None):
    """Router for loss functions."""
    if optimizer == 'adam':
        return tf.train.AdamOptimizer(lr).minimize(loss)
    elif optimizer == 'nadam':
        return tf.contrib.opt.NadamOptimizer(lr).minimize(loss)
    elif optimizer == 'sgd':
        return tf.train.GradientDescentOptimizer(lr).minimize(loss)
    elif optimizer == 'momentum':
        return momentum(loss=loss, lr=lr)
    elif optimizer == 'rmsprop':
        return tf.train.RMSPropOptimizer(lr).minimize(loss)
    elif optimizer == 'hessian':
        var_constraints = {model[k]: v for k, v in constraints.iteritems()}
        return tf.contrib.opt.ScipyOptimizerInterface(
            loss,
            options={'maxiter': 1000},
            var_to_bounds=var_constraints,
            method='fmin_tnc')
    else:
        raise RuntimeError('Cannot understand your loss function.')


def momentum(loss, lr, momentum=0.9):
    """Wrapper for SGD with momentum."""
    if isinstance(lr, dict):
        return tf.train.MomentumOptimizer(
            learning_rate=lr['learning_rate'],
            momentum=momentum).minimize(loss, lr['global_step'])
    else:
        return tf.train.MomentumOptimizer(lr, momentum=momentum).minimize(loss)


def derive_weights(weights, mu=0.15):
    """Derive class weights from a dictionary with a mu smoothing."""
    total = np.sum(weights.values())
    keys = weights.keys()
    class_weight = {}
    for k in keys:
        # score = math.log(mu * total / float(weights[k]))
        # class_weight[k] = score if score > 1.0 else 1.0
        class_weight[k] = total / float(weights[k])
    return class_weight.values()


def loss_interpreter(
        logits,
        labels,
        loss_type,
        weights=None,
        dataset_module=None):
    """Router for loss functions."""
    if loss_type is None:
        loss_type = dataset_module.default_loss_function
    if isinstance(weights, dict):
        weights = derive_weights(weights)
    if '@' in loss_type:
        # Temporary approach for multi-loss models
        # TODO use a similar approach as eval_metrics
        loss_types = loss_type.split('@')
        split_labels = tf.split(labels, len(loss_types), axis=-1)
        loss_list, score_list = [], []
        for idx, loss_type in enumerate(loss_types):
            if loss_type == 'cce':
                logit = tf.cast(logits[idx], tf.float32)
                label = tf.squeeze(tf.cast(split_labels[idx], tf.int64))
                it_loss, it_score = cce(
                    logits=logit,
                    labels=label,
                    weights=weights)
            elif loss_type == 'zero':
                it_loss, it_score = tf.constant(0.), tf.constant(0.)
            elif loss_type == 'tf_log_poisson':
                logit = tf.cast(logit, tf.float32)
                labels = tf.cast(labels, tf.float32)
                it_loss, it_score = tf_log_poisson(
                    logits=logit,
                    labels=split_labels[idx],
                    weights=weights)
            else:
                raise NotImplementedError
            loss_list += [it_loss]
            score_list += [it_score]
        return loss_list, score_list

    if loss_type == 'cce':
        logits = tf.cast(logits, tf.float32)
        labels = tf.squeeze(tf.cast(labels, tf.int64))
        return cce(
            logits=logits,
            labels=labels,
            weights=weights)
    elif loss_type == 'cce_ns':
        logits = tf.cast(logits, tf.float32)
        labels = tf.squeeze(tf.cast(labels, tf.int64))
        return cce_ns(
            logits=logits,
            labels=labels,
            weights=weights)
    elif loss_type == 'l2' or loss_type == 'L2':
        logits = tf.cast(logits, tf.float32)
        labels = tf.cast(labels, tf.float32)
        return l2(
            logits=logits,
            labels=labels,
            weights=weights)
    elif loss_type == 'log_loss':
        logits = tf.cast(logits, tf.float32)
        labels = tf.cast(labels, tf.float32)
        return log_loss(
            logits=logits,
            labels=labels,
            weights=weights)
    elif loss_type == 'huber':
        logits = tf.cast(logits, tf.float32)
        labels = tf.cast(labels, tf.float32)
        return huber(
            logits=logits,
            labels=labels,
            weights=weights)
    elif loss_type == 'sigmoid':
        logits = tf.cast(logits, tf.float32)
        labels = tf.cast(labels, tf.int64)
        return sigmoid_ce(
            logits=logits,
            labels=labels,
            weights=weights)
    elif loss_type == 'sigmoid_logits':
        logits = tf.cast(logits, tf.float32)
        labels = tf.cast(labels, tf.int64)
        return sigmoid_logit_ce(
            logits=logits,
            labels=labels,
            weights=weights)
    elif loss_type == 'pearson':
        logits = tf.cast(logits, tf.float32)
        labels = tf.cast(labels, tf.float32)
        return pearson_loss(
            logits=logits,
            labels=labels,
            weights=weights)
    elif loss_type == 'log_poisson':
        logits = tf.cast(logits, tf.float32)
        labels = tf.cast(labels, tf.float32)
        return log_poisson(
            logits=logits,
            labels=labels,
            weights=weights)
    elif loss_type == 'tf_log_poisson':
        logits = tf.cast(logits, tf.float32)
        labels = tf.cast(labels, tf.float32)
        return tf_log_poisson(
            logits=logits,
            labels=labels,
            weights=weights)
    else:
        raise RuntimeError('Cannot understand your loss function.')


def wd_loss(
        regularizations,
        loss,
        wd_penalty):
    """Calculate weight decay loss and add it to the main loss."""
    regs = []
    for k, v in regularizations.iteritems():
        lt = v['regularization_type']
        ls = v['regularization_strength']
        lw = v['weight']
        regs += [interpret_reg_loss(lw, lt) * ls]
    return loss + tf.add_n(regs)


def interpret_reg_loss(weight, loss_type):
    # TODO: Use Regularization class instead of this.
    if loss_type == 'l2' or loss_type == 'L2':
        return tf.nn.l2_loss(weight)
    elif loss_type == 'l1' or loss_type == 'L1':
        return tf.reduce_sum(tf.abs(weight))
    elif loss_type == 'frobenius':
        return frobenius(weight)
    elif loss_type == 'orthogonal':
        return orthogonal(weight)
    elif loss_type == 'laplace':
        return laplace(weight)
    else:
        raise NotImplementedError


def frobenius(x):
    """Norm on 2 - the trace of the correlation of x."""
    x_shape = [int(dx) for dx in x.get_shape()]
    conv_dim = np.prod(x_shape[:2])
    ch_dim = np.prod(x_shape[2:])
    x = tf.transpose(
        tf.reshape(
            tf.reshape(x, x_shape[:2] + [ch_dim]),
            [conv_dim, ch_dim]))
    mean_t = tf.reduce_mean(x, axis=1, keep_dims=True)
    cov_t = tf.matmul(
        (x - mean_t), tf.transpose(x - mean_t)) / (
            int(x.get_shape()[-1]) - 1)
    cov2_t = tf.diag(1. / tf.sqrt(tf.diag_part(cov_t)))
    cor = tf.matmul(tf.matmul(cov2_t, cov_t), cov2_t)
    return tf.trace((2 - cor) ** 2)


def laplace(x):
    """Regularize the laplace of the kernel."""
    kernel = np.asarray([
        [0.5, 1, 0.5],
        [1, -6, 1],
        [0.5, 1, 0.5]
    ])[:, :, None, None]
    kernel = np.repeat(
        kernel,
        int(x.get_shape()[-1]), axis=-2).astype(np.float32)
    tf_kernel = tf.get_variable(
        name='laplace_%s' % x.name.split('/')[-1].split(':')[0],
        initializer=kernel)
    reg_activity = tf.nn.conv2d(
        x,
        filter=tf_kernel,
        strides=[1, 1, 1, 1],
        padding='SAME')
    return tf.reduce_mean(tf.pow(reg_activity, 2))


def orthogonal(x, eps=1e-12):
    """Regularization for orthogonal components."""
    x_shape = [int(d) for d in x.get_shape()]
    out_rav_x = tf.reshape(tf.transpose(x, [3, 0, 1, 2]), [x_shape[3], -1])
    z = tf.matmul(out_rav_x, out_rav_x, transpose_b=True)  # Dot products
    x_norm = tf.norm(out_rav_x, axis=1, keep_dims=True)
    norm_kronecker = x_norm * tf.transpose(x_norm)  # kronecker prod of norms
    d = (z / norm_kronecker) ** 2  # Square so that minimum is orthogonal
    diag_d = tf.eye(x_shape[3]) * d
    return tf.reduce_mean(d - diag_d)  # Minimize off-diagonals


def cce(logits, labels, weights=None):
    """Sparse categorical cross entropy with weights."""
    if weights is not None:
        weights = tf.get_variable(
            name='weights', initializer=weights)[None, :]
        weights_per_label = tf.matmul(
            tf.one_hot(labels, 2), tf.transpose(tf.cast(weights, tf.float32)))
        return tf.reduce_mean(
            tf.multiply(
                weights_per_label,
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=logits, labels=labels))), tf.nn.softmax(logits)
    else:
        weights = 1.
        if len(logits.get_shape()) == len(labels.get_shape()):
            logits = tf.expand_dims(logits, axis=-1)
        return tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits,
                labels=labels) * weights), tf.nn.softmax(logits)


def cce_ns(logits, labels, weights=None):
    """Categorical cross entropy with weights."""
    if weights is not None:
        weights = tf.get_variable(
            name='weights', initializer=weights)[None, :]
        weights_per_label = tf.matmul(
            tf.one_hot(labels, 2), tf.transpose(tf.cast(weights, tf.float32)))
        return tf.reduce_mean(
            tf.multiply(
                weights_per_label,
                tf.nn.softmax_cross_entropy_with_logits(
                    logits=logits, labels=labels))), tf.nn.softmax(logits)
    else:
        weights = 1.
        return tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=logits,
                labels=labels) * weights), tf.nn.softmax(logits)


def l2(logits, labels, weights):
    """Wrapper for l2 loss."""
    delta = logits - labels
    if weights is not None:
        delta *= weights
    l2_loss = tf.nn.l2_loss(delta)
    return l2_loss, l2_loss


def l1(logits, labels, weights):
    """Wrapper for l2 loss."""
    delta = logits - labels
    if weights is not None:
        delta *= weights
    l1_loss = tf.reduce_sum(tf.abs(delta))
    return l1_loss, l1_loss


def huber(logits, labels, weights):
    """Wrapper for huber loss."""
    logits = tf.squeeze(logits)
    labels = tf.squeeze(labels)
    if weights is None:
        weights = 1.
    return tf.losses.huber_loss(
        predictions=logits,
        labels=labels,
        weights=weights), tf.nn.l2_loss(logits - labels)


def log_loss(logits, labels, weights):
    """Wrapper for log loss."""
    logits = tf.squeeze(logits)
    labels = tf.squeeze(labels)
    if weights is None:
        weights = 1.
    ll = tf.losses.log_loss(
        predictions=logits,
        labels=labels,
        weights=weights)
    return ll, ll


def log_poisson(logits, labels, weights, eps=1e-12):
    """Wrapper for log poisson loss from the Antilok dataset."""
    logits = tf.squeeze(logits)
    labels = tf.squeeze(labels)
    ll = tf.reduce_sum(logits) - tf.reduce_sum(
        tf.multiply(labels, tf.log(logits + eps)))
    return ll, ll


def tf_log_poisson(logits, labels, weights, eps=1e-12):
    """Wrapper for tensorflow log poisson loss."""
    logits = tf.squeeze(logits)
    labels = tf.squeeze(labels)
    ll = tf.reduce_mean(tf.nn.log_poisson_loss(
        targets=labels,
        log_input=logits))
    return ll, ll


def pearson_loss(logits, labels, weights):
    """Pearson dissimilarity loss."""
    rhos = 1 - pearson_score(pred=logits, labels=labels)
    mean_rhos = tf.reduce_mean(rhos)
    return mean_rhos, rhos


def sigmoid_logit_ce(logits, labels, weights, force_dtype=tf.float32):
    """Wrapper for sigmoid logit cross entropy loss."""
    if force_dtype:
        if logits.dtype != force_dtype:
            logits = tf.cast(logits, force_dtype)
        if labels.dtype != force_dtype:
            labels = tf.cast(labels, force_dtype)
    if weights is None:
        weights = 1.
    sig_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            labels=labels,
            logits=logits) * weights)
    return sig_loss, sig_loss


def sigmoid_ce(logits, labels, weights, force_dtype=tf.float32):
    """Wrapper for sigmoid cross entropy loss."""
    if force_dtype:
        if logits.dtype != force_dtype:
            logits = tf.cast(logits, force_dtype)
        if labels.dtype != force_dtype:
            labels = tf.cast(labels, force_dtype)
    if weights is None:
        weights = 1.

    # Note logits are have a sigmoid activation
    sig_loss = labels * -tf.log(logits) + (1 - labels) * -tf.log(1 - logits)
    sig_loss *= weights
    sig_loss = tf.reduce_mean(sig_loss)
    return sig_loss, sig_loss


def create_lr_schedule(train_batch, num_training):
    """Create a step-wise decaying learning rate schedule."""
    batch_denom = num_training * train_batch
    learning_rate_fn = learning_rate_with_decay(
        batch_size=train_batch,
        batch_denom=batch_denom,  # * 10 # 256,
        num_images=num_training,
        boundary_epochs=[30, 60, 80, 90],
        decay_rates=[1, 0.1, 0.01, 0.001, 1e-4])
    global_step = tf.train.get_or_create_global_step()
    learning_rate = learning_rate_fn(global_step)

    # Create a tensor named learning_rate for logging purposes
    tf.summary.scalar(
        'learning_rate',
        tf.identity(learning_rate, name='learning_rate'))
    return {'learning_rate': learning_rate, 'global_step': global_step}


def learning_rate_with_decay(
        batch_size,
        batch_denom,
        num_images,
        boundary_epochs,
        decay_rates):
    """Get a learning rate that decays step-wise as training progresses.

    Args:
        batch_size: the number of examples processed in each training batch.
        batch_denom: this value will be used to scale the base learning rate.
            `0.1 * batch size` is divided by this number, such that when
            batch_denom == batch_size, the initial learning rate will be 0.1.
        num_images: total number of images that will be used for training.
        boundary_epochs: list of ints representing the epochs at which we
            decay the learning rate.
        decay_rates: list of floats representing the decay rates to be used
            for scaling the learning rate. Should be the same length as
            boundary_epochs.

    Returns:
        Returns a function that takes a single argument - the number of batches
        trained so far (global_step)- and returns the learning rate to be used
        for training the next batch.
    """
    initial_learning_rate = 0.1 * batch_size / batch_denom
    batches_per_epoch = num_images / batch_size

    # Multiply the learning rate by 0.1 at 100, 150, and 200 epochs.
    boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
    vals = [initial_learning_rate * decay for decay in decay_rates]

    def learning_rate_fn(global_step):
        global_step = tf.cast(global_step, tf.int32)
        return tf.train.piecewise_constant(global_step, boundaries, vals)

    return learning_rate_fn
