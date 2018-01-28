import tensorflow as tf


def metric_interpreter(metric, pred, labels):
    """Interpret string as a metric."""
    if isinstance(pred, list):
        scores = []
        for idx, pd in enumerate(pred):
            label = labels[:, idx]
            if metric == 'accuracy':
                scores += [class_accuracy(
                    pred=pd,
                    labels=label)]
            elif metric == 'argmax_softmax_accuracy':
                final_dim = len(pd.get_shape()) - 1
                proc_pred = tf.cast(
                    tf.argmax(tf.nn.softmax(pd, dim=-1), axis=final_dim),
                    tf.float32)
                label = tf.cast(label, tf.float32)
                scores += [tf.reduce_mean(
                    tf.cast(
                        tf.equal(
                            proc_pred, label), tf.float32))]
            else:
                raise NotImplementedError
        return scores
    else:
        if metric == 'accuracy':
            return class_accuracy(
                pred=pred,
                labels=labels)
        elif metric == 'mean':
            return mean_score(
                pred=pred,
                labels=labels)
        elif metric == 'l2':
            return l2_score(
                pred=pred,
                labels=labels)
        elif metric == 'pearson':
            return pearson_summary_score(
                pred=pred,
                labels=labels)
        elif metric == 'pearson_columnwise':
            return pearson_summary_score(
                pred=pred,
                labels=labels)
        elif metric == 'l2_columnwise':
            return l2_columnwise(
                pred=pred,
                labels=labels)
        elif metric == 'sigmoid_pearson':
            return pearson_summary_score(
                pred=tf.nn.sigmoid(pred),
                labels=labels)
        elif metric == 'round_pearson':
            return pearson_summary_score(
                pred=tf.squeeze(tf.round(pred)),
                labels=tf.squeeze(labels))
        elif metric == 'sigmoid_accuracy':
            return sigmoid_accuracy(
                pred=pred,
                labels=labels)
        elif metric == 'argmax_softmax_accuracy':
            final_dim = len(pred.get_shape()) - 1
            proc_pred = tf.cast(
                tf.argmax(tf.nn.softmax(pred, dim=-1), axis=final_dim),
                tf.float32)
            label = tf.cast(labels, tf.float32)
            return tf.reduce_mean(
                 tf.cast(
                     tf.equal(
                         proc_pred, label), tf.float32))
        elif metric == 'f1':
            return f1(
                pred=pred,
                labels=labels)
        elif metric == 'precision':
            return precision(
                pred=pred,
                labels=labels)
        elif metric == 'recall':
            return recall(
                pred=pred,
                labels=labels)
        elif metric == 'auc':
            return auc(
                pred=pred,
                labels=labels)
        else:
            raise RuntimeError('Cannot understand the dataset metric.')


def tabulate_confusion_metrics(pred, labels, force_dtype=tf.float32):
    pred = tf.round(tf.nn.sigmoid(pred))
    if force_dtype:
        if pred.dtype != force_dtype:
            pred = tf.cast(pred, force_dtype)
        if labels.dtype != force_dtype:
            labels = tf.cast(labels, force_dtype)
    tp = tf.count_nonzero(
        pred * labels,
        axis=1,
        dtype=force_dtype)
    tn = tf.count_nonzero(
        (pred - 1) * (labels - 1),
        axis=1,
        dtype=force_dtype)
    fp = tf.count_nonzero(
        pred * (labels - 1),
        axis=1,
        dtype=force_dtype)
    fn = tf.count_nonzero(
        (pred - 1) * labels,
        axis=1,
        dtype=force_dtype)
    return tp, tn, fp, fn


def precision(pred, labels, tp=None, tn=None, fp=None, fn=None, eps=1e-12):
    if tp is None or tn is None or fp is None or fn is None:
        tp, tn, fp, fn = tabulate_confusion_metrics(pred, labels)
    return tp / (tp + fp + eps)


def recall(pred, labels, tp=None, tn=None, fp=None, fn=None, eps=1e-12):
    if tp is None or tn is None or fp is None or fn is None:
        tp, tn, fp, fn = tabulate_confusion_metrics(pred, labels)
    return tp / (tp + fn + eps)


def f1(pred, labels, force_dtype=tf.float32, eps=1e-12):
    """Calculate F1 score."""
    tp, tn, fp, fn = tabulate_confusion_metrics(pred, labels)
    p = precision(None, None, tp, tn, fp, fn)
    r = recall(None, None, tp, tn, fp, fn)
    f1 = (2 * p * r) / (p + r + eps)
    return tf.reduce_mean(f1)


def auc(pred, labels, force_dtype=tf.float32):
    if force_dtype:
        if pred.dtype != force_dtype:
            pred = tf.cast(pred, force_dtype)
        if labels.dtype != force_dtype:
            labels = tf.cast(labels, force_dtype)
    adj_pred = tf.round(tf.nn.sigmoid(pred))
    auc, _ = tf.metrics.auc(
        predictions=adj_pred,
        labels=labels)
    return auc


def sigmoid_accuracy(pred, labels, force_dtype=tf.float32):
    """Apply sigmoid to pred and round output."""
    if force_dtype:
        if pred.dtype != force_dtype:
            pred = tf.cast(pred, force_dtype)
        if labels.dtype != force_dtype:
            labels = tf.cast(labels, force_dtype)
    adj_pred = tf.round(tf.nn.sigmoid(pred))
    return tf.reduce_mean(tf.cast(tf.equal(adj_pred, labels), tf.float32))


def mean_score(pred, labels, force_dtype=tf.float32):
    """Accuracy with integer type."""
    if force_dtype:
        if pred.dtype != force_dtype:
            pred = tf.cast(pred, force_dtype)
        if labels.dtype != force_dtype:
            labels = tf.cast(labels, force_dtype)
    return tf.reduce_mean(tf.cast(tf.equal(pred, labels), tf.float32))


def class_accuracy(pred, labels):
    """Accuracy of 1/n*sum(pred_i == label_i)."""
    return tf.reduce_mean(
        tf.to_float(tf.equal(tf.argmax(pred, 1), tf.cast(
            labels, dtype=tf.int64))))


def l2_score(pred, labels):
    """L2 distance."""
    return tf.nn.l2_loss(pred - labels)


def tf_confusion_matrix(pred, labels):
    """Wrapper for calculating confusion matrix."""
    return tf.contrib.metrics.confusion_matrix(pred, labels)


def pearson_summary_score(pred, labels, eps_1=1e-4, eps_2=1e-12):
    """Wrapper to summarize a vector of correlations."""
    return tf.reduce_mean(
        pearson_score(
            pred,
            labels,
            eps_1=eps_1,
            eps_2=eps_2))


def pearson_columnwise(pred, labels, eps_1=1e-4, eps_2=1e-12):
    """Wrapper to gather per-column pearson correlations."""
    return pearson_score(
            pred,
            labels,
            eps_1=eps_1,
            eps_2=eps_2)


def l2_columnwise(pred, labels, eps_1=1e-4, eps_2=1e-12):
    """Wrapper to gather per-column l2 distance."""
    d = (pred - labels) ** 2
    return tf.sqrt(tf.reduce_mean(d, axis=0))


def pearson_score(pred, labels, eps_1=1e-4, eps_2=1e-12):
    """Pearson loss function."""
    x_shape = [int(x) for x in pred.get_shape()]
    y_shape = [int(x) for x in labels.get_shape()]
    if x_shape[-1] == 1 and len(x_shape) == 2:
        # If calculating score across exemplars
        pred = tf.squeeze(pred)
        x_shape = [x_shape[0]]
        labels = tf.squeeze(labels)
        y_shape = [y_shape[0]]

    if len(x_shape) > 2:
        # Reshape tensors
        x1_flat = tf.contrib.layers.flatten(pred)
    else:
        # Squeeze off singletons to make x1/x2 consistent
        x1_flat = tf.squeeze(pred)
    if len(y_shape) > 2:
        x2_flat = tf.contrib.layers.flatten(labels)
    else:
        x2_flat = tf.squeeze(labels)
    x1_mean = tf.reduce_mean(x1_flat, keep_dims=True, axis=[-1]) + eps_1
    x2_mean = tf.reduce_mean(x2_flat, keep_dims=True, axis=[-1]) + eps_1

    x1_flat_normed = x1_flat - x1_mean
    x2_flat_normed = x2_flat - x2_mean

    count = int(x2_flat.get_shape()[-1])
    cov = tf.div(
        tf.reduce_sum(
            tf.multiply(
                x1_flat_normed, x2_flat_normed),
            -1),
        count)
    x1_std = tf.sqrt(
        tf.div(
            tf.reduce_sum(
                tf.square(x1_flat - x1_mean),
                -1),
            count))
    x2_std = tf.sqrt(
        tf.div(
            tf.reduce_sum(
                tf.square(x2_flat - x2_mean),
                -1),
            count))
    return cov / (tf.multiply(x1_std, x2_std) + eps_2)
