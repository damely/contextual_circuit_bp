import tensorflow as tf


class activations(object):
    """Wrapper class for activation functions."""
    def __getitem__(self, name):
        return getattr(self, name)

    def __contains__(self, name):
        return hasattr(self, name)

    def __init__(self, kwargs=None):
        """Globals for activation functions."""
        self.training = None
        if kwargs is not None:
            self.update_params(**kwargs)

    def update_params(self, **kwargs):
        for k, v in kwargs.iteritems():
            setattr(self, k, v)

    def relu(self, x, **kwargs):
        """Rectified linearity."""
        return tf.nn.relu(x)

    def selu(self, x, **kwargs):
        """Scaled exponential linear unit."""
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale * tf.where(
            tf.greater_equal(x, 0.0), x, alpha * tf.nn.elu(x))

    def soft_plus(self, x, **kwargs):
        """Soft logistic."""
        return tf.nn.softplus(x)

    def crelu(self, x, **kwargs):
        """Concatenated +/- relu."""
        return tf.nn.crelu(x)

    def logistic(self, x, coef=1.0, t=0.0, **kwargs):
        """Antilok et al., 2016 logistic activation function."""
        return coef * tf.log(1 + tf.exp(coef * (x - t)))

    def sigmoid(self, x, **kwargs):
        """Elementwise sigmoid."""
        return tf.sigmoid(x)

    def round(self, x, **kwargs):
        """Elementwise round."""
        return tf.round(x)

    def softmax(self, x, **kwargs):
        """Elementwise softmax."""
        return tf.nn.softmax(x, axis=-1)

    def square(self, x, **kwargs):
        """Elementwise square."""
        return tf.pow(x, 2)

