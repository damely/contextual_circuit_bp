"""Pooling wrapper. TODO: add interpreter."""
import tensorflow as tf


class pool(object):
    """Wrapper class for pooling operations."""

    def __getitem__(self, name):
        """Get attribute from class."""
        return getattr(self, name)

    def __contains__(self, name):
        """Check if class contains attribute."""
        return hasattr(self, name)

    def __init__(self, kwargs=None):
        """Global variables for pooling functions."""
        self.k = [1, 2, 2, 1]  # kernel size
        self.s = [1, 2, 2, 1]  # stride
        self.p = 'SAME'  # padding
        self.update_params(kwargs)

    def update_params(self, kwargs):
        """Update the class attributes with kwargs."""
        if kwargs is not None:
            for k, v in kwargs.iteritems():
                setattr(self, k, v)

    def interpret_1dpool(
            self,
            context,
            bottom,
            name,
            filter_size,
            stride_size,
            pool_type,
            padding=None,
            kwargs=None):
        """Apply the appropriate 2D pooling type."""
        if filter_size is None:
            filter_size = self.k
        if stride_size is None:
            stride_size = self.s
        if padding is None:
            padding = self.p
        if kwargs is not None:
            self.update_params(kwargs)
        if len(bottom.get_shape()) < 3:
            bottom = tf.expand_dims(bottom, axis=-1)
        if pool_type == 'max':
            context, act = max_pool_1d(
                self=context,
                bottom=bottom,
                name=name,
                k=filter_size,
                s=stride_size,
                p=padding)
        elif pool_type == 'avg':
            context, act = avg_pool_1d(
                self=context,
                bottom=bottom,
                name=name,
                k=filter_size,
                s=stride_size,
                p=padding)
        else:
            raise RuntimeError('Cannot understand specifed pool type.')
        return context, act

    def interpret_2dpool(
            self,
            context,
            bottom,
            name,
            filter_size,
            stride_size,
            pool_type,
            padding=None,
            kwargs=None):
        """Apply the appropriate 2D pooling type."""
        if filter_size is None:
            filter_size = self.k
        if stride_size is None:
            stride_size = self.s
        if padding is None:
            padding = self.p
        if kwargs is not None:
            self.update_params(kwargs)
        if pool_type == 'max':
            context, act = max_pool(
                self=context,
                bottom=bottom,
                name=name,
                k=filter_size,
                s=stride_size,
                p=padding)
        elif pool_type == 'avg' or pool_type == 'mean':
            context, act = avg_pool(
                self=context,
                bottom=bottom,
                name=name,
                k=filter_size,
                s=stride_size,
                p=padding)
        else:
            raise RuntimeError('Cannot understand specifed pool type.')
        return context, act

    def interpret_3dpool(
            self,
            context,
            bottom,
            name,
            filter_size,
            stride_size,
            pool_type,
            padding=None,
            kwargs=None):
        """Apply the appropriate 3D pooling type."""
        if filter_size is None:
            filter_size = self.k
        if stride_size is None:
            stride_size = self.s
        if padding is None:
            padding = self.p
        if kwargs is not None:
            self.update_params(kwargs)
        if pool_type == 'max':
            context, act = max_pool_3d(
                self=context,
                bottom=bottom,
                name=name,
                k=filter_size,
                s=stride_size,
                p=padding)
        elif pool_type == 'avg':
            context, act = avg_pool_3d(
                self=context,
                bottom=bottom,
                name=name,
                k=filter_size,
                s=stride_size,
                p=padding)
        else:
            raise RuntimeError('Cannot understand specifed pool type.')
        return context, act


def avg_pool(
        self,
        bottom,
        name,
        k=[1, 2, 2, 1],
        s=[1, 2, 2, 1],
        p='SAME'):
    """Local average pooling."""
    return self, tf.nn.avg_pool(
        bottom,
        ksize=k,
        strides=s,
        padding=p,
        name=name)


def max_pool(
        self,
        bottom,
        name,
        k=[1, 2, 2, 1],
        s=[1, 2, 2, 1],
        p='SAME'):
    """Local max pooling."""
    return self, tf.nn.max_pool(
        bottom,
        ksize=k,
        strides=s,
        padding=p,
        name=name)


def avg_pool_1d(
        self,
        bottom,
        name,
        k=[2],
        s=[2],
        pooling_type='AVG',
        p='SAME'):
    """Local 1d avg pooling."""
    return self, tf.squeeze(tf.nn.pool(
        input=bottom,
        window_shape=k,
        pooling_type=pooling_type,
        padding=p,
        strides=s,
        name=name))


def max_pool_1d(
        self,
        bottom,
        name,
        k=[2],
        s=[2],
        pooling_type='MAX',
        p='SAME'):
    """Local 1d max pooling."""
    return self, tf.squeeze(tf.nn.pool(
        input=bottom,
        window_shape=k,
        pooling_type=pooling_type,
        padding=p,
        strides=s,
        name=name))


def avg_pool_3d(
        self,
        bottom,
        name,
        k=[1, 2, 2, 1],
        s=[1, 2, 2, 1],
        p='SAME'):
    """Spatiotemporal average pooling."""
    return self, tf.nn.avg_pool3d(
        bottom,
        ksize=k,
        strides=s,
        padding=p,
        name=name)


def max_pool_3d(
        self,
        bottom,
        name,
        k=[1, 2, 2, 2, 1],  # D/H/W
        s=[1, 2, 2, 2, 1],  # D/H/W
        p='SAME'):
    """Spatiotemporal max pooling."""
    return self, tf.nn.max_pool3d(
        bottom,
        ksize=k,
        strides=s,
        padding=p,
        name=name)
