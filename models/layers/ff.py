"""Functions for handling feedforward and pooling operations."""
import numpy as np
import tensorflow as tf
from ops import initialization
from models.layers.activations import activations
from models.layers.normalizations import normalizations
from models.layers.ff_functions import ff as ff_fun
from models.layers import pool


class ff(object):
    """Wrapper class for network filter operations."""

    def __getitem__(self, name):
        """Get attribute from class."""
        return getattr(self, name)

    def __contains__(self, name):
        """Check if class contains attribute."""
        return hasattr(self, name)

    def __init__(self, kwargs=None):
        """Global variables for ff functions."""
        self.update_params(kwargs)
        self.pool_class = pool.pool()

    def update_params(self, kwargs):
        """Update the class attributes with kwargs."""
        if kwargs is not None:
            for k, v in kwargs.iteritems():
                setattr(self, k, v)

    def gather(
            self,
            context,
            act,
            in_channels,
            out_channels,
            filter_size,
            name,
            it_dict):
        """Layer that gathers a value at an index."""
        context, act = gather_value_layer(
            self=context,
            bottom=act,
            aux=it_dict['aux'],
            name=name)
        return context, act

    def bias(
            self,
            context,
            act,
            in_channels,
            out_channels,
            filter_size,
            name,
            it_dict):
        """Add a learnable bias layer."""
        context, act = bias_layer(
            self=context,
            bottom=act,
            in_channels=in_channels,
            out_channels=out_channels,
            name=name,
            filter_size=filter_size)
        return context, act

    def dog(
            self,
            context,
            act,
            in_channels,
            out_channels,
            filter_size,
            name,
            it_dict):
        """Add a Difference of Gaussians layer."""
        context, act = dog_layer(
            self=context,
            bottom=act,
            layer_weights=out_channels,
            name=name)
        return context, act

    def DoG(
            self,
            context,
            act,
            in_channels,
            out_channels,
            filter_size,
            name,
            it_dict):
        """Add a Difference of Gaussians layer."""
        context, act = dog_layer(
            self=context,
            bottom=act,
            layer_weights=out_channels,
            name=name)
        return context, act

    def dog_conv(
            self,
            context,
            act,
            in_channels,
            out_channels,
            filter_size,
            name,
            it_dict):
        """Add a Difference of Gaussians layer."""
        context, act = dog_conv_layer(
            self=context,
            bottom=act,
            layer_weights=out_channels,
            name=name)
        return context, act

    def gabor_conv(
            self,
            context,
            act,
            in_channels,
            out_channels,
            filter_size,
            name,
            it_dict):
        """Add a Difference of Gaussians layer."""
        context, act = gabor_conv_layer(
            self=context,
            bottom=act,
            layer_weights=out_channels,
            name=name)
        return context, act

    def conv(
            self,
            context,
            act,
            in_channels,
            out_channels,
            filter_size,
            name,
            it_dict):
        """Add a 2D convolution layer."""
        context, act = conv_layer(
            self=context,
            bottom=act,
            in_channels=in_channels,
            out_channels=out_channels,
            name=name,
            filter_size=filter_size)
        return context, act

    def alexnet_conv(
            self,
            context,
            act,
            in_channels,
            out_channels,
            filter_size,
            name,
            it_dict):
        """Add a 2D convolution layer."""
        context, act = alexnet_conv_layer(
            self=context,
            bottom=act,
            in_channels=in_channels,
            out_channels=out_channels,
            name=name,
            filter_size=filter_size,
            aux=it_dict)
        return context, act

    def conv3d(
            self,
            context,
            act,
            in_channels,
            out_channels,
            filter_size,
            name,
            it_dict):
        """Add a 3D convolution layer."""
        context, act = conv3d_layer(
            self=context,
            bottom=act,
            in_channels=in_channels,
            out_channels=out_channels,
            name=name,
            filter_size=filter_size,
            aux=it_dict)
        return context, act

    def sep_conv(
            self,
            context,
            act,
            in_channels,
            out_channels,
            filter_size,
            name,
            it_dict):
        """Add a separable 2D convolution layer."""
        context, act = sep_conv_layer(
            self=context,
            bottom=act,
            in_channels=in_channels,
            out_channels=out_channels,
            name=name,
            filter_size=filter_size)
        return context, act

    def time_sep_conv3d(
            self,
            context,
            act,
            in_channels,
            out_channels,
            filter_size,
            name,
            it_dict):
        """Add a separable 3D convolution layer."""
        context, act = time_sep_conv3d_layer(
            self=context,
            bottom=act,
            in_channels=in_channels,
            out_channels=out_channels,
            name=name,
            filter_size=filter_size,
            aux=it_dict)
        return context, act

    def complete_sep_conv3d(
            self,
            context,
            act,
            in_channels,
            out_channels,
            filter_size,
            name,
            it_dict):
        """Add a separable 3D convolution layer."""
        context, act = complete_sep_conv3d_layer(
            self=context,
            bottom=act,
            in_channels=in_channels,
            out_channels=out_channels,
            name=name,
            filter_size=filter_size,
            aux=it_dict)
        return context, act

    def lstm2d(
            self,
            context,
            act,
            in_channels,
            out_channels,
            filter_size,
            name,
            it_dict):
        """Convolutional LSTM."""
        context, act = lstm2d_layer(
            self=context,
            bottom=act,
            in_channels=in_channels,
            out_channels=out_channels,
            name=name,
            filter_size=filter_size)
        return context, act

    def sgru2d(
            self,
            context,
            act,
            in_channels,
            out_channels,
            filter_size,
            name,
            it_dict):
        """Convolutional separable GRU."""
        context, act = sgru2d_layer(
            self=context,
            bottom=act,
            in_channels=in_channels,
            out_channels=out_channels,
            name=name,
            filter_size=filter_size,
            aux=it_dict)
        return context, act

    def gru2d(
            self,
            context,
            act,
            in_channels,
            out_channels,
            filter_size,
            name,
            it_dict):
        """Convolutional GRU."""
        context, act = gru2d_layer(
            self=context,
            bottom=act,
            in_channels=in_channels,
            out_channels=out_channels,
            name=name,
            filter_size=filter_size,
            aux=it_dict)
        return context, act

    def sepgru2d(
            self,
            context,
            act,
            in_channels,
            out_channels,
            filter_size,
            name,
            it_dict):
        """Convolutional GRU."""
        context, act = sepgru2d_layer(
            self=context,
            bottom=act,
            in_channels=in_channels,
            out_channels=out_channels,
            name=name,
            filter_size=filter_size,
            aux=it_dict)
        return context, act

    def alexnet_sepgru2d(
            self,
            context,
            act,
            in_channels,
            out_channels,
            filter_size,
            name,
            it_dict):
        """Convolutional GRU w alexnet FF drive."""
        context, act = alexnet_sepgru2d_layer(
            self=context,
            bottom=act,
            in_channels=in_channels,
            out_channels=out_channels,
            name=name,
            filter_size=filter_size,
            aux=it_dict)
        return context, act

    def mru2d(
            self,
            context,
            act,
            in_channels,
            out_channels,
            filter_size,
            name,
            it_dict):
        """Convolutional MRU."""
        context, act = mru2d_layer(
            self=context,
            bottom=act,
            in_channels=in_channels,
            out_channels=out_channels,
            name=name,
            filter_size=filter_size,
            aux=it_dict)
        return context, act

    def rnn2d(
            self,
            context,
            act,
            in_channels,
            out_channels,
            filter_size,
            name,
            it_dict):
        """Convolutional RNN."""
        context, act = rnn2d_layer(
            self=context,
            bottom=act,
            in_channels=in_channels,
            out_channels=out_channels,
            name=name,
            filter_size=filter_size,
            aux=it_dict)
        return context, act

    def lstm1d(
            self,
            context,
            act,
            in_channels,
            out_channels,
            filter_size,
            name,
            it_dict):
        """1d LSTM."""
        context, act = lstm1d_layer(
            self=context,
            bottom=act,
            in_channels=in_channels,
            out_channels=out_channels,
            name=name,
            filter_size=filter_size,
            aux=it_dict)
        return context, act

    def gru1d(
            self,
            context,
            act,
            in_channels,
            out_channels,
            filter_size,
            name,
            it_dict):
        """1d GRU."""
        context, act = gru1d_layer(
            self=context,
            bottom=act,
            in_channels=in_channels,
            out_channels=out_channels,
            name=name,
            filter_size=filter_size,
            aux=it_dict)
        return context, act

    def conv1d(
            self,
            context,
            act,
            in_channels,
            out_channels,
            filter_size,
            name,
            it_dict):
        """1d conv."""
        context, act = conv1d_layer(
            self=context,
            bottom=act,
            in_channels=in_channels,
            out_channels=out_channels,
            name=name,
            filter_size=filter_size)
        return context, act

    def fc(
            self,
            context,
            act,
            in_channels,
            out_channels,
            filter_size,
            name,
            it_dict):
        """Add a fully-connected layer."""
        context, act = fc_layer(
            self=context,
            bottom=act,
            in_channels=in_channels,
            out_channels=out_channels,
            name=name)
        return context, act

    def multi_fc(
            self,
            context,
            act,
            in_channels,
            out_channels,
            filter_size,
            name,
            it_dict):
        """Add multiple fully-connected layers."""
        context, act = multi_fc_layer(
            self=context,
            bottom=act,
            in_channels=in_channels,
            out_channels=out_channels,
            name=name)
        return context, act

    def sparse_pool(
            self,
            context,
            act,
            in_channels,
            out_channels,
            filter_size,
            name,
            it_dict):
        """Add a sparse pooling layer."""
        context, act = sparse_pool_layer(
            self=context,
            bottom=act,
            in_channels=in_channels,
            out_channels=out_channels,
            aux=it_dict,
            name=name)
        return context, act

    def res(
            self,
            context,
            act,
            in_channels,
            out_channels,
            filter_size,
            name,
            it_dict):
        """Add a residual layer."""
        context, act = resnet_layer(
            self=context,
            bottom=act,
            aux=it_dict['aux'],
            layer_weights=it_dict['weights'],
            name=name)
        return context, act

    def _pass(
            self,
            context,
            act,
            in_channels,
            out_channels,
            filter_size,
            name,
            it_dict):
        """Skip a filter operation on this layer."""
        return context, act

    def pool(
            self,
            context,
            act,
            in_channels,
            out_channels,
            filter_size,
            name,
            it_dict):
        """Wrapper for 2d pool. TODO: add op flexibility."""
        if filter_size is None:
            filter_size = [1, 2, 2, 1]
        stride_size = it_dict.get('stride', [1, 2, 2, 1])
        if not isinstance(filter_size, list):
            filter_size = [1, filter_size, filter_size, 1]
        if not isinstance(stride_size, list):
            filter_size = [1, stride_size, stride_size, 1]
        if 'aux' in it_dict and 'pool_type' in it_dict['aux']:
            pool_type = it_dict['aux']['pool_type']
        else:
            pool_type = 'max'

        context, act = self.pool_class.interpret_2dpool(
            context=context,
            bottom=act,
            name=name,
            filter_size=filter_size,
            stride_size=stride_size,
            pool_type=pool_type
        )
        return context, act

    def pool1d(
            self,
            context,
            act,
            in_channels,
            out_channels,
            filter_size,
            name,
            it_dict):
        """Wrapper for 1d pool."""
        if filter_size is None:
            filter_size = [2]
        stride_size = it_dict.get('stride', [2])
        if not isinstance(filter_size, list):
            filter_size = [filter_size]
        if not isinstance(stride_size, list):
            filter_size = [stride_size]
        if 'aux' in it_dict and 'pool_type' in it_dict['aux']:
            pool_type = it_dict['aux']['pool_type']
        else:
            pool_type = 'max'

        context, act = self.pool_class.interpret_1dpool(
            context=context,
            bottom=act,
            name=name,
            filter_size=filter_size,
            stride_size=stride_size,
            pool_type=pool_type
        )
        return context, act

    def pool3d(
            self,
            context,
            act,
            in_channels,
            out_channels,
            filter_size,
            name,
            it_dict):
        """Wrapper for 3d pool. TODO: add op flexibility."""
        if filter_size is None:
            filter_size = [1, 2, 2, 2, 1]
        stride_size = it_dict.get('stride', [1, 2, 2, 2, 1])
        if not isinstance(filter_size, list):
            filter_size = [1, filter_size, filter_size, filter_size, 1]
        if not isinstance(stride_size, list):
            filter_size = [1, stride_size, stride_size, stride_size, 1]
        if 'aux' in it_dict and 'pool_type' in it_dict['aux']:
            pool_type = it_dict['aux']['pool_type']
        else:
            pool_type = 'max'

        context, act = self.pool_class.interpret_3dpool(
            context=context,
            bottom=act,
            name=name,
            filter_size=filter_size,
            stride_size=stride_size,
            pool_type=pool_type
        )
        return context, act

    def vgg16(
            self,
            context,
            act,
            in_channels,
            out_channels,
            filter_size,
            name,
            it_dict):
        """Wrapper for loading an imagnet initialized VGG16."""
        context, act = ff_fun.vgg16(
            self=context,
            bottom=act,
            aux=it_dict['aux'],
            layer_weights=it_dict['weights'],
            name=name)
        return context, act


def gather_value_layer(
        self,
        bottom,
        aux,
        name):
    """Gather a value from a location in an activity tensor."""
    assert aux is not None,\
        'Gather op needs an aux dict with h/w coordinates.'
    assert 'h' in aux.keys() and 'w' in aux.keys(),\
        'Gather op dict needs h/w key value pairs'
    h = aux['h']
    w = aux['w']
    out = tf.squeeze(bottom[:, h, w, :])
    return self, out


# TODO: move each of these ops into a script in the functions folder.
def dog_layer(
        self,
        bottom,
        layer_weights,
        name,
        init_weight=10.,
        model_dtype=tf.float32):
    """Antilok et al 2016 difference of gaussians."""
    tshape = [int(x) for x in bottom.get_shape()]  # Flatten input
    rows = tshape[0]
    cols = np.prod(tshape[1:])
    flat_bottom = tf.reshape(bottom, [rows, cols])
    hw = tshape[1:3][::-1]
    min_dim = np.min(hw)
    act_size = [int(x) for x in flat_bottom.get_shape()]  # Use flattened input
    len_act = len(act_size)
    assert len_act == 2, 'DoG layer needs 2D matrix not %sD tensor.' % len_act
    grid_xx, grid_yy = tf.meshgrid(
        tf.range(hw[0]),
        tf.range(hw[1]))
    grid_xx = tf.cast(grid_xx, tf.float32)
    grid_yy = tf.cast(grid_yy, tf.float32)
    pi = tf.constant(np.pi, dtype=model_dtype)

    def randomize_init(
            bounds,
            layer_weights,
            d1=4.0,
            d2=2.0,
            dtype=np.float32):
        """Initialize starting positions of DoG parameters as uniform rand."""
        init_dict = {}
        for k, v in bounds.iteritems():
            it_inits = []
            r = v[1] - v[0]
            for idx in range(layer_weights):
                it_inits += [v[0] + (r / d1) + np.random.rand() * (r / d2)]
            init_dict[k] = np.asarray(it_inits, dtype=dtype)
        return init_dict

    def DoG(bottom, x, y, sc, ss, rc, rs):
        """DoG operation."""
        pos = ((grid_xx - x)**2 + (grid_yy - y)**2)
        center = tf.exp(-pos / 2 / sc) / (2 * (sc) * pi)
        surround = tf.exp(-pos / 2 / (sc + ss)) / (2 * (sc + ss) * pi)
        weight_vec = tf.reshape((rc * (center)) - (rs * (surround)), [-1, 1])
        return tf.matmul(bottom, weight_vec), weight_vec

    if isinstance(layer_weights, list):
        layer_weights = layer_weights[0]

    # Construct model bounds
    bounds = {
        'x_pos': [
            0.,
            hw[0],
        ],
        'y_pos': [
            0.,
            hw[1],
        ],
        'size_center': [
            0.1,
            min_dim,
        ],
        'size_surround': [
            0.,
            min_dim,
        ],
        'center_weight': [
            0.,
            init_weight,
        ],
        'surround_weight': [
            0.,
            init_weight,
        ],
    }

    # Create tensorflow weights
    initializers = randomize_init(
        bounds=bounds,
        layer_weights=layer_weights)
    lgn_x = tf.get_variable(
        name='%s_x_pos' % name,
        dtype=model_dtype,
        initializer=initializers['x_pos'],
        trainable=True)
    lgn_y = tf.get_variable(
        name='%s_y_pos' % name,
        dtype=model_dtype,
        initializer=initializers['y_pos'],
        trainable=True)
    lgn_sc = tf.get_variable(
        name='%s_size_center' % name,
        dtype=model_dtype,
        initializer=initializers['size_center'],
        trainable=True)
    lgn_ss = tf.get_variable(
        name='%s_size_surround' % name,
        dtype=model_dtype,
        initializer=initializers['size_surround'],
        trainable=True)
    lgn_rc = tf.get_variable(
        name='%s_center_weight' % name,
        dtype=model_dtype,
        initializer=initializers['center_weight'],
        trainable=True)
    lgn_rs = tf.get_variable(
        name='%s_surround_weight' % name,
        dtype=model_dtype,
        initializer=initializers['surround_weight'],
        trainable=True)

    output, dog_weights = [], []
    for i in range(layer_weights):
        activities, weight_vec = DoG(
            bottom=flat_bottom,
            x=lgn_x[i],
            y=lgn_y[i],
            sc=lgn_sc[i],
            ss=lgn_ss[i],
            rc=lgn_rc[i],
            rs=lgn_rs[i])
        output += [activities]
        dog_weights += [weight_vec]
    self.var_dict[('%s_weights' % name, 0)] = dog_weights
    flat_bottom = tf.reshape(bottom, [rows, cols])
    return self, tf.concat(axis=1, values=output)


def resnet_layer(
        self,
        bottom,
        layer_weights,
        name,
        aux=None,
        combination=None):  # tf.multiply
    """Residual layer."""
    ln = '%s_branch' % name
    rlayer = tf.identity(bottom)
    if aux is not None:
        if 'activation' in aux.keys():
            activation = aux['activation']
        if 'normalization' in aux.keys():
            normalization = aux['normalization']
        if 'combination' in aux.keys():
            if aux['combination'] == 'add':
                combination = tf.add
            elif aux['combination'] == 'prod':
                combination = tf.multiply
            elif aux['combination'] == 'add_prod':
                combination = lambda x, y: tf.concat(
                        tf.add(x, y),
                        tf.multiply(x, y)
                    )
        else:
            combination = tf.add
    if normalization is not None:
        if normalization is not 'batch':
            raise RuntimeError(
                'Normalization not yet implemented for non-batchnorm.')
        nm = normalizations({'training': self.training})[normalization]
    else:
        nm = lambda x: x
    if activation is not None:
        ac = activations()[activation]
    else:
        ac = lambda x: x
    for idx, lw in enumerate(layer_weights):
        ln = '%s_%s' % (name, idx)
        self, rlayer = conv_layer(
            self=self,
            bottom=rlayer,
            in_channels=int(rlayer.get_shape()[-1]),
            out_channels=lw,
            name=ln)
        rlayer = nm(ac(rlayer), None, None, None)
        if isinstance(rlayer, tuple):
            rlayer = rlayer[0]
    return self, combination(rlayer, bottom)


def conv_layer(
        self,
        bottom,
        out_channels,
        name,
        in_channels=None,
        filter_size=3,
        stride=[1, 1, 1, 1],
        padding='SAME'):
    """2D convolutional layer."""
    with tf.variable_scope(name):
        if in_channels is None:
            in_channels = int(bottom.get_shape()[-1])
        self, filt, conv_biases = get_conv_var(
            self=self,
            filter_size=filter_size,
            in_channels=in_channels,
            out_channels=out_channels,
            name=name)
        conv = tf.nn.conv2d(bottom, filt, stride, padding=padding)
        bias = tf.nn.bias_add(conv, conv_biases)
        return self, bias


def alexnet_conv_layer(
        self,
        bottom,
        out_channels,
        name,
        in_channels=None,
        filter_size=3,
        stride=[1, 1, 1, 1],
        padding='SAME',
        aux=None):
    """2D convolutional layer."""
    assert aux is not None, 'Pass the location of alexnet weights.'
    assert 'alexnet_npy' in aux.keys(), 'Pass an alexnet_npy key.'
    alexnet_weights = np.load(aux['alexnet_npy']).item()
    alexnet_key = aux['alexnet_layer']
    alexnet_filter, alexnet_bias = alexnet_weights[alexnet_key]
    with tf.variable_scope(name):
        if in_channels is None:
            in_channels = int(bottom.get_shape()[-1])
        assert out_channels == alexnet_filter.shape[-1],\
            'Set weights = %s.' % alexnet_filter.shape[-1]
        if in_channels < alexnet_filter.shape[-2] and in_channels == 1:
            alexnet_filter = np.expand_dims(alexnet_filter[:, :, 0, :], axis=2)
        elif in_channels < alexnet_filter.shape[-2]:
            raise RuntimeError('Input features = %s, Alexnet features = %s' % (
                in_channels, alexnet_filter.shape[-2]))
        self, filters = get_var(
            self=self,
            initial_value=alexnet_filter,
            name=name,
            idx=0,
            var_name=name + "_filters")
        self, biases = get_var(
            self=self,
            initial_value=alexnet_bias,
            name=name,
            idx=1,
            var_name=name + "_biases")
        conv = tf.nn.conv2d(bottom, filters, stride, padding=padding)
        bias = tf.nn.bias_add(conv, biases)
        return self, bias


def conv1d_layer(
        self,
        bottom,
        out_channels,
        name,
        in_channels=None,
        filter_size=3,
        stride=[1, 1, 1, 1],
        padding='SAME'):
    """1D convolutional layer."""
    with tf.variable_scope(name):
        if in_channels is None:
            in_channels = int(bottom.get_shape()[-1])
        bottom = tf.expand_dims(
            tf.expand_dims(bottom, axis=-1),
            axis=1)
        self, filt, conv_biases = get_conv_var(
            self=self,
            filter_size=filter_size,
            in_channels=1,
            out_channels=out_channels,
            name=name,
            kernel=[filter_size])
        filt = tf.expand_dims(filt, 0)
        conv = tf.nn.conv2d(bottom, filt, stride, padding=padding)
        bias = tf.squeeze(tf.nn.bias_add(conv, conv_biases), axis=1)
        return self, bias


def bias_layer(
        self,
        bottom,
        out_channels,
        name,
        in_channels=None,
        filter_size=1,
        padding='SAME'):
    """2D convolutional layer."""
    with tf.variable_scope(name):
        if in_channels is None:
            in_channels = int(bottom.get_shape()[-1])
        self, _, conv_biases = get_conv_var(
            self=self,
            filter_size=filter_size,
            in_channels=in_channels,
            out_channels=in_channels,
            name=name)
        bias = tf.nn.bias_add(bottom, conv_biases)
        return self, bias


def dog_conv_layer(
        self,
        bottom,
        out_channels,
        name,
        in_channels=None,
        filter_size=3,
        stride=[1, 1, 1, 1],
        padding='SAME'):
    """2D convolutional layer. NOT IMPLEMENTED."""
    with tf.variable_scope(name):
        if in_channels is None:
            in_channels = int(bottom.get_shape()[-1])
        self, filt, conv_biases = get_conv_var(
            self=self,
            filter_size=filter_size,
            in_channels=in_channels,
            out_channels=out_channels,
            name=name)
        conv = tf.nn.conv2d(bottom, filt, stride, padding=padding)
        bias = tf.nn.bias_add(conv, conv_biases)
        return self, bias


def gabor_conv_layer(
        self,
        bottom,
        out_channels,
        name,
        in_channels=None,
        filter_size=3,
        stride=[1, 1, 1, 1],
        padding='SAME'):
    """2D convolutional layer. NOT IMPLEMENTED."""
    with tf.variable_scope(name):
        if in_channels is None:
            in_channels = int(bottom.get_shape()[-1])
        self, filt, conv_biases = get_conv_var(
            self=self,
            filter_size=filter_size,
            in_channels=in_channels,
            out_channels=out_channels,
            name=name)
        conv = tf.nn.conv2d(bottom, filt, stride, padding=padding)
        bias = tf.nn.bias_add(conv, conv_biases)
        return self, bias


def sep_conv_layer(
        self,
        bottom,
        out_channels,
        name,
        in_channels=None,
        filter_size=3,
        stride=[1, 1, 1, 1],
        padding='SAME',
        multiplier=1,
        aux=None):
    """2D convolutional layer."""
    if aux is not None and 'ff_aux' in aux.keys():
        if 'multiplier' in aux['ff_aux']:
            multiplier = aux['multiplier']
    with tf.variable_scope(name):
        if in_channels is None:
            in_channels = int(bottom.get_shape()[-1])
        self, dfilt, _ = get_conv_var(
            self=self,
            filter_size=filter_size,
            in_channels=in_channels,
            out_channels=multiplier,
            name='d_%s' % name)
        self, pfilt, conv_biases = get_conv_var(
            self=self,
            filter_size=1,
            in_channels=multiplier,
            out_channels=out_channels,
            name='p_%s' % name)
        conv = tf.nn.separable_conv2d(
            input=bottom,
            depthwise_filter=dfilt,
            pointwise_filt=pfilt,
            strides=stride,
            padding=padding)
        bias = tf.nn.bias_add(conv, conv_biases)
        return self, bias


def time_sep_conv3d_layer(
        self,
        bottom,
        out_channels,
        name,
        in_channels=None,
        filter_size=3,
        stride=[1, 1, 1, 1, 1],
        padding='SAME',
        aux=None):
    """3D convolutional layer."""
    with tf.variable_scope(name):
        if in_channels is None:
            in_channels = int(bottom.get_shape()[-1])
        timesteps = int(bottom.get_shape()[1])

        # T/H/W/In/Out
        # 1. Time convolution
        t_kernel = [timesteps, 1, 1]
        self, t_filt, conv_biases = get_conv_var(
            self=self,
            filter_size=filter_size,
            in_channels=in_channels,
            out_channels=out_channels,
            name='%s_t' % name,
            kernel=t_kernel)
        t_conv = tf.nn.conv3d(
            bottom,
            t_filt,
            stride,
            padding=padding)

        # 1b. Add nonlinearity between separable convolutions
        if aux is not None and 'ff_aux' in aux.keys():
            if 'activation' in aux['ff_aux']:
                t_conv = activations()[aux['ff_aux']['activation']](t_conv)

        # 2. HW Convolution
        hwk_kernel = [1, filter_size, filter_size]
        self, hwk_filt, _ = get_conv_var(
            self=self,
            filter_size=filter_size,
            in_channels=out_channels,
            out_channels=out_channels,
            name='%s_hw' % name,
            kernel=hwk_kernel)
        hwk_conv = tf.nn.conv3d(
            t_conv,
            hwk_filt,
            stride,
            padding=padding)
        bias = tf.nn.bias_add(hwk_conv, conv_biases)
        return self, bias


def complete_sep_conv3d_layer(
        self,
        bottom,
        out_channels,
        name,
        in_channels=None,
        filter_size=3,
        stride2d=[1, 1, 1, 1],
        stride3d=[1, 1, 1, 1, 1],
        padding='SAME',
        multiplier=1,
        aux=None):
    """3D convolutional layer."""
    with tf.variable_scope(name):
        if in_channels is None:
            in_channels = int(bottom.get_shape()[-1])
        timesteps = int(bottom.get_shape()[1])

        # T/H/W/In/Out
        # 1. Time convolution
        t_kernel = [timesteps, 1, 1]
        self, t_filt, conv_biases = get_conv_var(
            self=self,
            filter_size=filter_size,
            in_channels=in_channels,
            out_channels=out_channels,
            name='%s_t' % name,
            kernel=t_kernel)
        t_conv = tf.nn.conv3d(
            bottom,
            t_filt,
            stride3d,
            padding=padding)

        # 1b. Add nonlinearity between separable convolutions
        if aux is not None and 'ff_aux' in aux.keys():
            if 'activation' in aux['ff_aux']:
                t_conv = activations()[aux['ff_aux']['activation']](t_conv)

        # 2. Sep Convolution shared across timepoints
        self, dfilt, _ = get_conv_var(
            self=self,
            filter_size=filter_size,
            in_channels=out_channels,
            out_channels=multiplier,
            name='d_%s' % name)
        self, pfilt, conv_biases = get_conv_var(
            self=self,
            filter_size=1,
            in_channels=out_channels * multiplier,
            out_channels=out_channels,
            name='p_%s' % name)

        # Inefficient. TODO: Develop the C++ code for this
        t_bottom = tf.split(t_conv, timesteps, axis=1)
        t_convs = []
        for ts in range(timesteps):
            t_convs += [tf.expand_dims(
                    tf.nn.separable_conv2d(
                        input=tf.squeeze(t_bottom[ts], axis=1),
                        depthwise_filter=dfilt,
                        pointwise_filter=pfilt,
                        strides=stride2d,
                        padding=padding),
                    axis=1)]
        t_convs = tf.concat(t_convs, axis=1)
        bias = tf.nn.bias_add(t_conv, conv_biases)

        return self, bias


def lstm1d_layer(
        self,
        bottom,
        out_channels,
        name,
        in_channels=None,
        filter_size=3,
        gate_filter_size=1,
        aux=None):
    """1D LSTM convolutional layer."""

    def lstm_condition(
            step,
            timesteps,
            split_bottom,
            h,
            x_gate_filters,
            h_gate_filters,
            gate_biases):
        """Condition for ending LSTM."""
        return step < timesteps

    def lstm_body(
            step,
            timesteps,
            split_bottom,
            h,
            x_gate_filters,
            h_gate_filters,
            gate_biases):
        """Calculate updates for 1d lstm."""

        # Concatenate X_t and the hidden state
        X = tf.gather(split_bottom, step)
        if isinstance(h, list):
            h_prev = tf.gather(h, step)
        else:
            h_prev = h

        # Perform matmul
        x_gate_convs = tf.matmul(X, x_gate_filters)
        h_gate_convs = tf.matmul(h_prev, h_gate_filters)

        # Calculate gates
        gate_activites = x_gate_convs + h_gate_convs + gate_biases

        # Reshape and split into appropriate gates
        gate_sizes = [int(x) for x in gate_activites.get_shape()]
        div_g = gate_sizes[:-1] + [gate_sizes[-1] // 4, 4]
        res_gates = tf.reshape(
                gate_activites,
                div_g)
        split_gates = tf.split(res_gates, 4, axis=2)
        f, i, o, c = split_gates
        f = tf.squeeze(gate_nl(f))
        i = tf.squeeze(gate_nl(i))
        o = tf.squeeze(gate_nl(o))
        c = tf.squeeze(cell_nl(c))
        c_update = (f * h) + (c * i)
        if isinstance(h, list):
            # If we are storing the hidden state at each step
            h[step] = o * c_update
        else:
            # If we are only keeping the final hidden state
            h = o * c_update
        step += 1
        return (
                step,
                timesteps,
                split_bottom,
                h,
                x_gate_filters,
                h_gate_filters,
                gate_biases
                )

    # Scope the 1d lstm
    with tf.variable_scope(name):
        if in_channels is None:
            in_channels = int(bottom.get_shape()[-1])
        timesteps = int(bottom.get_shape()[1])

        if aux is not None and 'gate_nl' in aux.keys():
            gate_nl = aux['gate_nl']
        else:
            gate_nl = tf.sigmoid

        if aux is not None and 'cell_nl' in aux.keys():
            cell_nl = aux['cell_nl']
        else:
            cell_nl = tf.nn.relu

        if aux is not None and 'random_init' in aux.keys():
            random_init = aux['random_init']
        else:
            random_init = True

        # LSTM: pack i/o/f/c gates into a single tensor
        # X_facing tensor, H_facing tensor for both weights and biases
        x_weights, h_weights = [], []
        biases = []
        gates = ['f', 'i', 'o', 'c']
        filter_sizes = len(gates) * [filter_size]
        for idx, (g, fs) in enumerate(zip(gates, filter_sizes)):
            self, iW, ib = get_fc_var(
                self=self,
                in_size=in_channels,
                out_size=out_channels,
                name='%s_X_gate_%s' % (name, g))
            x_weights += [iW]
            biases += [ib]
            self, iW, ib = get_fc_var(
                self=self,
                in_size=out_channels,
                out_size=out_channels,
                name='%s_H_gate_%s' % (name, g))
            h_weights += [iW]

        # Concatenate each into 2d tensors
        x_gate_filters = tf.concat(x_weights, axis=-1)
        h_gate_filters = tf.concat(h_weights, axis=-1)
        gate_biases = tf.concat(biases, axis=0)

        # Split bottom up by timesteps and initialize cell and hidden states
        split_bottom = tf.split(bottom, timesteps, axis=1)
        split_bottom = [tf.squeeze(x, axis=1) for x in split_bottom]  # Time
        h_size = [
            int(x) for x in split_bottom[0].get_shape()[:-1]] + [out_channels]
        if aux is not None and 'ff_aux' in aux.keys():
            if 'store_hidden_states' in aux['ff_aux']:
                if random_init:
                    hidden_state = [
                        tf.random_normal(
                            shape=h_size,
                            mean=0.0,
                            stddev=0.1) for x in range(timesteps)]
                else:
                    hidden_state = [
                        tf.zeros_like(split_bottom[0])
                        for x in range(timesteps)]
        else:
            if random_init:
                hidden_state = tf.random_normal(
                    shape=h_size,
                    mean=0.0,
                    stddev=0.1)
            else:
                hidden_state = tf.zeros_like(h_size)

        # While loop
        step = tf.constant(0)  # timestep iterator
        elems = [
            step,
            timesteps,
            split_bottom,
            hidden_state,
            x_gate_filters,
            h_gate_filters,
            gate_biases
        ]

        if aux is not None and 'ff_aux' in aux.keys():
            if 'swap_memory' in aux['ff_aux']:
                swap_memory = aux['ff_aux']['swap_memory']
        else:
            swap_memory = True
        returned = tf.while_loop(
            lstm_condition,
            lstm_body,
            loop_vars=elems,
            back_prop=True,
            swap_memory=swap_memory)

        # Prepare output
        _, _, _, h_updated, _, _, _ = returned

        # Save input/hidden facing weights
        return self, h_updated


def gru1d_layer(
        self,
        bottom,
        out_channels,
        name,
        in_channels=None,
        filter_size=3,
        gate_filter_size=1,
        aux=None):
    """1D GRU convolutional layer."""

    def gru_condition(
            step,
            timesteps,
            split_bottom,
            h,
            x_gate_filters,
            h_gate_filters,
            x_filter,
            h_filter,
            gate_biases,
            h_bias):
        """Condition for ending gru."""
        return step < timesteps

    def gru_body(
            step,
            timesteps,
            split_bottom,
            h,
            x_gate_filters,
            h_gate_filters,
            x_filter,
            h_filter,
            gate_biases,
            h_bias):
        """Calculate updates for 1d gru."""

        # Concatenate X_t and the hidden state
        X = tf.gather(split_bottom, step)
        if isinstance(h, list):
            h_prev = tf.gather(h, step)
        else:
            h_prev = h

        # Perform matmul
        x_gate_convs = tf.matmul(X, x_gate_filters)
        h_gate_convs = tf.matmul(h_prev, h_gate_filters)

        # Calculate gates
        gate_activites = x_gate_convs + h_gate_convs + gate_biases
        nl_activities = gate_nl(gate_activites)

        # Reshape and split into appropriate gates
        gate_sizes = [int(x) for x in nl_activities.get_shape()]
        div_g = gate_sizes[:-1] + [gate_sizes[-1] // 2, 2]
        res_gates = tf.reshape(
                nl_activities,
                div_g)
        z, r = tf.split(res_gates, 2, axis=2)

        # Update drives
        h_update = tf.squeeze(r) * h_prev

        # Perform matmul
        x_convs = tf.matmul(X, x_filter)
        h_convs = tf.matmul(h_update, h_filter)

        # Integrate circuit
        z = tf.squeeze(z)
        h_update = (z * h_prev) + ((1 - z) * cell_nl(
            x_convs + h_convs + h_bias))
        if isinstance(h, list):
            # If we are storing the hidden state at each step
            h[step] = h_update
        else:
            # If we are only keeping the final hidden state
            h = h_update
        step += 1
        return (
            step,
            timesteps,
            split_bottom,
            h,
            x_gate_filters,
            h_gate_filters,
            x_filter,
            h_filter,
            gate_biases,
            h_bias
        )

    # Scope the 1d gru
    with tf.variable_scope(name):
        if in_channels is None:
            in_channels = int(bottom.get_shape()[-1])
        timesteps = int(bottom.get_shape()[1])

        if aux is not None and 'gate_nl' in aux.keys():
            gate_nl = aux['gate_nl']
        else:
            gate_nl = tf.sigmoid

        if aux is not None and 'cell_nl' in aux.keys():
            cell_nl = aux['cell_nl']
        else:
            cell_nl = tf.nn.relu

        if aux is not None and 'random_init' in aux.keys():
            random_init = aux['random_init']
        else:
            random_init = True

        # LSTM: pack i/o/f/c gates into a single tensor
        # X_facing tensor, H_facing tensor for both weights and biases
        x_weights, h_weights = [], []
        biases = []
        gates = ['z', 'r']
        filter_sizes = len(gates) * [filter_size]
        for idx, (g, fs) in enumerate(zip(gates, filter_sizes)):
            self, iW, ib = get_fc_var(
                self=self,
                in_size=in_channels,
                out_size=out_channels,
                name='%s_X_gate_%s' % (name, g))
            x_weights += [iW]
            biases += [ib]
            self, iW, _ = get_fc_var(
                self=self,
                in_size=out_channels,
                out_size=out_channels,
                name='%s_H_gate_%s' % (name, g))
            h_weights += [iW]

        # Concatenate each into 2d tensors
        x_gate_filters = tf.concat(x_weights, axis=-1)
        h_gate_filters = tf.concat(h_weights, axis=-1)
        gate_biases = tf.concat(biases, axis=0)

        # Split off last h weight
        self, x_filter, _ = get_fc_var(
            self=self,
            in_size=in_channels,
            out_size=out_channels,
            name='%s_X_gate_%s' % (name, 'x'))
        self, h_filter, h_bias = get_fc_var(
            self=self,
            in_size=out_channels,
            out_size=out_channels,
            name='%s_H_gate_%s' % (name, 'h'))

        # Split bottom up by timesteps and initialize cell and hidden states
        split_bottom = tf.split(bottom, timesteps, axis=1)
        split_bottom = [tf.squeeze(x, axis=1) for x in split_bottom]  # Time
        h_size = [
            int(x) for x in split_bottom[0].get_shape()[:-1]] + [out_channels]
        if aux is not None and 'ff_aux' in aux.keys():
            if 'store_hidden_states' in aux['ff_aux']:
                if random_init:
                    hidden_state = [
                        tf.random_normal(
                            shape=h_size,
                            mean=0.0,
                            stddev=0.1) for x in range(timesteps)]
                else:
                    hidden_state = [
                        tf.zeros_like(split_bottom[0])
                        for x in range(timesteps)]
        else:
            if random_init:
                hidden_state = tf.random_normal(
                    shape=h_size,
                    mean=0.0,
                    stddev=0.1)
            else:
                hidden_state = tf.zeros_like(h_size)

        # While loop
        step = tf.constant(0)  # timestep iterator
        elems = [
            step,
            timesteps,
            split_bottom,
            hidden_state,
            x_gate_filters,
            h_gate_filters,
            x_filter,
            h_filter,
            gate_biases,
            h_bias
        ]

        if aux is not None and 'ff_aux' in aux.keys():
            if 'swap_memory' in aux['ff_aux']:
                swap_memory = aux['ff_aux']['swap_memory']
        else:
            swap_memory = True
        returned = tf.while_loop(
            gru_condition,
            gru_body,
            loop_vars=elems,
            back_prop=True,
            swap_memory=swap_memory)

        # Prepare output
        h_updated = returned[3]

        # Save input/hidden facing weights
        return self, h_updated


def lstm2d_layer(
        self,
        bottom,
        out_channels,
        name,
        in_channels=None,
        filter_size=3,
        gate_filter_size=1,
        aux=None):
    """2D LSTM convolutional layer."""

    def lstm_condition(
            step,
            timesteps,
            split_bottom,
            h,
            x_gate_filters,
            h_gate_filters,
            gate_biases):
        """Condition for ending LSTM."""
        return step < timesteps

    def lstm_body(
            step,
            timesteps,
            split_bottom,
            h,
            x_gate_filters,
            h_gate_filters,
            gate_biases):
        """Calculate updates for 2d lstm."""

        # Concatenate X_t and the hidden state
        X = tf.gather(split_bottom, step)
        if isinstance(h, list):
            h_prev = tf.gather(h, step)
        else:
            h_prev = h

        # Perform convolutions
        x_gate_convs = tf.nn.conv2d(
            X,
            x_gate_filters,
            [1, 1, 1, 1],
            padding='SAME')
        h_gate_convs = tf.nn.conv2d(
            h_prev,
            h_gate_filters,
            [1, 1, 1, 1],
            padding='SAME')

        # Calculate gates
        gate_activites = x_gate_convs + h_gate_convs + gate_biases

        # Reshape and split into appropriate gates
        gate_sizes = [int(x) for x in gate_activites.get_shape()]
        div_g = gate_sizes[:-1] + [gate_sizes[-1] // 4, 4]
        res_gates = tf.reshape(
                gate_activites,
                div_g)
        split_gates = tf.split(res_gates, 4, axis=4)
        f, i, o, c = split_gates
        f = tf.squeeze(gate_nl(f))
        i = tf.squeeze(gate_nl(i))
        o = tf.squeeze(gate_nl(o))
        c = tf.squeeze(cell_nl(c))
        c_update = (f * h) + (c * i)
        if isinstance(h, list):
            # If we are storing the hidden state at each step
            h[step] = o * c_update
        else:
            # If we are only keeping the final hidden state
            h = o * c_update
        step += 1
        return (
                step,
                timesteps,
                split_bottom,
                h,
                x_gate_filters,
                h_gate_filters,
                gate_biases
                )

    # Scope the 2d lstm
    with tf.variable_scope(name):
        if in_channels is None:
            in_channels = int(bottom.get_shape()[-1])
        timesteps = int(bottom.get_shape()[1])

        if aux is not None and 'gate_nl' in aux.keys():
            gate_nl = aux['gate_nl']
        else:
            gate_nl = tf.sigmoid

        if aux is not None and 'cell_nl' in aux.keys():
            cell_nl = aux['cell_nl']
        else:
            cell_nl = tf.nn.relu

        if aux is not None and 'random_init' in aux.keys():
            random_init = aux['random_init']
        else:
            random_init = True

        # LSTM: pack i/o/f/c gates into a single tensor
        # X_facing tensor, H_facing tensor for both weights and biases
        x_weights, h_weights = [], []
        biases = []
        gates = ['f', 'i', 'o', 'c']
        filter_sizes = len(gates) * [filter_size]
        for idx, (g, fs) in enumerate(zip(gates, filter_sizes)):
            self, iW, ib = get_conv_var(
                self=self,
                filter_size=fs,
                in_channels=in_channels,  # For the hidden state
                out_channels=out_channels,
                name='%s_X_gate_%s' % (name, g))
            x_weights += [iW]
            biases += [ib]
            self, iW, ib = get_conv_var(
                self=self,
                filter_size=fs,
                in_channels=out_channels,  # For the hidden state
                out_channels=out_channels,
                name='%s_H_gate_%s' % (name, g))
            h_weights += [iW]

        # Concatenate each into 3d tensors
        x_gate_filters = tf.concat(x_weights, axis=-1)
        h_gate_filters = tf.concat(h_weights, axis=-1)
        gate_biases = tf.concat(biases, axis=0)

        # Split bottom up by timesteps and initialize cell and hidden states
        split_bottom = tf.split(bottom, timesteps, axis=1)
        split_bottom = [tf.squeeze(x, axis=1) for x in split_bottom]  # Time
        h_size = [
            int(x) for x in split_bottom[0].get_shape()[:-1]] + [out_channels]
        if aux is not None and 'ff_aux' in aux.keys():
            if 'store_hidden_states' in aux['ff_aux']:
                if random_init:
                    hidden_state = [
                        tf.random_normal(
                            shape=h_size,
                            mean=0.0,
                            stddev=0.1) for x in range(timesteps)]
                else:
                    hidden_state = [
                        tf.zeros_like(split_bottom[0])
                        for x in range(timesteps)]
        else:
            if random_init:
                hidden_state = tf.random_normal(
                    shape=h_size,
                    mean=0.0,
                    stddev=0.1)
            else:
                hidden_state = tf.zeros_like(h_size)

        # While loop
        step = tf.constant(0)  # timestep iterator
        elems = [
            step,
            timesteps,
            split_bottom,
            hidden_state,
            x_gate_filters,
            h_gate_filters,
            gate_biases
        ]

        if aux is not None and 'ff_aux' in aux.keys():
            if 'swap_memory' in aux['ff_aux']:
                swap_memory = aux['ff_aux']['swap_memory']
        else:
            swap_memory = True
        returned = tf.while_loop(
            lstm_condition,
            lstm_body,
            loop_vars=elems,
            back_prop=True,
            swap_memory=swap_memory)

        # Prepare output
        _, _, _, h_updated, _, _, _ = returned

        # Save input/hidden facing weights
        return self, h_updated


def sepgru2d_layer(
        self,
        bottom,
        out_channels,
        name,
        in_channels=None,
        filter_size=3,
        gate_filter_size=1,
        aux=None):
    """2D separable GRU convolutional layer."""
    def sepgru_condition(
            step,
            timesteps,
            split_bottom,
            h,
            x_z_depth_filters,
            x_z_point_filters,
            h_z_depth_filters,
            h_z_point_filters,
            h_z_point_bias,
            x_r_depth_filters,
            x_r_point_filters,
            h_r_depth_filters,
            h_r_point_filters,
            h_r_point_bias,
            x_depth_filter,
            x_point_filter,
            h_depth_filter,
            h_point_filter,
            h_bias):
        """Condition for ending GRU."""
        return step < timesteps

    def sepgru_body(
            step,
            timesteps,
            split_bottom,
            h,
            x_z_depth_filters,
            x_z_point_filters,
            h_z_depth_filters,
            h_z_point_filters,
            h_z_point_bias,
            x_r_depth_filters,
            x_r_point_filters,
            h_r_depth_filters,
            h_r_point_filters,
            h_r_point_bias,
            x_depth_filter,
            x_point_filter,
            h_depth_filter,
            h_point_filter,
            h_bias):
        """Calculate updates for 2D separable GRU."""
        # Concatenate X_t and the hidden state
        X = tf.gather(split_bottom, step)
        if isinstance(h, list):
            h_prev = tf.gather(h, step)
        else:
            h_prev = h

        # Perform gate convolutions
        x_z = tf.nn.separable_conv2d(
            input=X,
            depthwise_filter=x_z_depth_filters,
            pointwise_filter=x_z_point_filters,
            strides=[1, 1, 1, 1],
            padding='SAME')
        x_r = tf.nn.separable_conv2d(
            input=X,
            depthwise_filter=x_r_depth_filters,
            pointwise_filter=x_r_point_filters,
            strides=[1, 1, 1, 1],
            padding='SAME')
        h_z = tf.nn.separable_conv2d(
            input=h_prev,
            depthwise_filter=h_z_depth_filters,
            pointwise_filter=h_z_point_filters,
            strides=[1, 1, 1, 1],
            padding='SAME')
        h_r = tf.nn.separable_conv2d(
            input=h_prev,
            depthwise_filter=h_r_depth_filters,
            pointwise_filter=h_r_point_filters,
            strides=[1, 1, 1, 1],
            padding='SAME')

        # Calculate gates
        z = gate_nl(x_z + h_z + h_z_point_bias)
        r = gate_nl(x_r + h_r + h_r_point_bias)

        # Update drives
        h_update = tf.squeeze(r) * h_prev

        # Perform FF/REC convolutions
        x_convs = tf.nn.separable_conv2d(
            input=X,
            depthwise_filter=x_depth_filter,
            pointwise_filter=x_point_filter,
            strides=[1, 1, 1, 1],
            padding='SAME')
        h_convs = tf.nn.separable_conv2d(
            input=h_update,
            depthwise_filter=h_depth_filter,
            pointwise_filter=h_point_filter,
            strides=[1, 1, 1, 1],
            padding='SAME')

        # Integrate circuit
        z = tf.squeeze(z)
        h_update = (z * h_prev) + ((1 - z) * cell_nl(
            x_convs + h_convs + h_bias))
        if isinstance(h, list):
            # If we are storing the hidden state at each step
            h[step] = h_update
        else:
            # If we are only keeping the final hidden state
            h = h_update
        step += 1
        return (
            step,
            timesteps,
            split_bottom,
            h,
            x_z_depth_filters,
            x_z_point_filters,
            h_z_depth_filters,
            h_z_point_filters,
            h_z_point_bias,
            x_r_depth_filters,
            x_r_point_filters,
            h_r_depth_filters,
            h_r_point_filters,
            h_r_point_bias,
            x_depth_filter,
            x_point_filter,
            h_depth_filter,
            h_point_filter,
            h_bias)

    # Scope the 2D GRU
    with tf.variable_scope(name):
        if in_channels is None:
            in_channels = int(bottom.get_shape()[-1])
        timesteps = int(bottom.get_shape()[1])

        if aux is not None and 'gate_nl' in aux.keys():
            gate_nl = aux['gate_nl']
        else:
            gate_nl = tf.sigmoid

        if aux is not None and 'cell_nl' in aux.keys():
            cell_nl = aux['cell_nl']
        else:
            cell_nl = tf.nn.relu

        if aux is not None and 'random_init' in aux.keys():
            random_init = aux['random_init']
        else:
            random_init = True

        # GRU: pack z/r/h gates into a single tensor
        # X_facing tensor, H_facing tensor for both weights and biases
        channel_multiplier = 1

        # Z
        self, x_z_depth_filters, _ = get_conv_var(
            self=self,
            filter_size=gate_filter_size,
            in_channels=in_channels,
            out_channels=channel_multiplier,
            name='%s_X_gate_depth_%s' % (name, 'z'))
        self, x_z_point_filters, _ = get_conv_var(
            self=self,
            filter_size=1,
            in_channels=in_channels * channel_multiplier,
            out_channels=out_channels,
            name='%s_X_gate_point_%s' % (name, 'z'))
        self, h_z_depth_filters, _ = get_conv_var(
            self=self,
            filter_size=gate_filter_size,
            in_channels=out_channels,  # For the hidden state
            out_channels=channel_multiplier,
            name='%s_H_gate_depth_%s' % (name, 'z'))
        self, h_z_point_filters, h_z_point_bias = get_conv_var(
            self=self,
            filter_size=1,
            in_channels=out_channels * channel_multiplier,
            out_channels=out_channels,
            name='%s_H_gate_point_%s' % (name, 'z'))

        # R
        self, x_r_depth_filters, _ = get_conv_var(
            self=self,
            filter_size=gate_filter_size,
            in_channels=in_channels,
            out_channels=channel_multiplier,
            name='%s_X_gate_depth_%s' % (name, 'r'))
        self, x_r_point_filters, _ = get_conv_var(
            self=self,
            filter_size=1,
            in_channels=in_channels * channel_multiplier,
            out_channels=out_channels,
            name='%s_X_gate_point_%s' % (name, 'r'))
        self, h_r_depth_filters, _ = get_conv_var(
            self=self,
            filter_size=gate_filter_size,
            in_channels=out_channels,  # For the hidden state
            out_channels=channel_multiplier,
            name='%s_H_gate_depth_%s' % (name, 'r'))
        self, h_r_point_filters, h_r_point_bias = get_conv_var(
            self=self,
            filter_size=1,
            in_channels=out_channels * channel_multiplier,
            out_channels=out_channels,
            name='%s_H_gate_point_%s' % (name, 'r'))

        # Split off last h weight
        self, h_depth_filter, h_bias = get_conv_var(
            self=self,
            filter_size=filter_size,
            in_channels=out_channels,  # For the hidden state
            out_channels=channel_multiplier,
            name='%s_H_gate_depth_%s' % (name, 'h'))
        self, h_point_filter, h_bias = get_conv_var(
            self=self,
            filter_size=1,
            in_channels=out_channels * channel_multiplier,
            out_channels=out_channels,
            name='%s_H_gate_point_%s' % (name, 'h'))
        self, x_depth_filter, _ = get_conv_var(
            self=self,
            filter_size=filter_size,
            in_channels=in_channels,  # For the hidden state
            out_channels=channel_multiplier,
            name='%s_X_gate_depth_%s' % (name, 'x'))
        self, x_point_filter, _ = get_conv_var(
            self=self,
            filter_size=1,
            in_channels=in_channels * channel_multiplier,
            out_channels=out_channels,
            name='%s_X_gate_point_%s' % (name, 'x'))

        # Split bottom up by timesteps and initialize cell and hidden states
        split_bottom = tf.split(bottom, timesteps, axis=1)
        split_bottom = [tf.squeeze(x, axis=1) for x in split_bottom]  # Time
        h_size = [
            int(x) for x in split_bottom[0].get_shape()[:-1]] + [out_channels]
        if aux is not None and 'ff_aux' in aux.keys():
            if 'store_hidden_states' in aux['ff_aux']:
                if random_init:
                    hidden_state = [
                        tf.random_normal(
                            shape=h_size,
                            mean=0.0,
                            stddev=0.1) for x in range(timesteps)]
                else:
                    hidden_state = [
                        tf.zeros_like(split_bottom[0])
                        for x in range(timesteps)]
        else:
            if random_init:
                hidden_state = tf.random_normal(
                    shape=h_size,
                    mean=0.0,
                    stddev=0.1)
            else:
                hidden_state = tf.zeros_like(h_size)

        # While loop
        step = tf.constant(0)  # timestep iterator
        elems = [
            step,
            timesteps,
            split_bottom,
            hidden_state,
            x_z_depth_filters,
            x_z_point_filters,
            h_z_depth_filters,
            h_z_point_filters,
            h_z_point_bias,
            x_r_depth_filters,
            x_r_point_filters,
            h_r_depth_filters,
            h_r_point_filters,
            h_r_point_bias,
            x_depth_filter,
            x_point_filter,
            h_depth_filter,
            h_point_filter,
            h_bias
        ]

        if aux is not None and 'ff_aux' in aux.keys():
            if 'swap_memory' in aux['ff_aux']:
                swap_memory = aux['ff_aux']['swap_memory']
        else:
            swap_memory = True
        returned = tf.while_loop(
            sepgru_condition,
            sepgru_body,
            loop_vars=elems,
            back_prop=True,
            swap_memory=swap_memory)

        # Prepare output
        h_updated = returned[3]
        return self, h_updated


def alexnet_sepgru2d_layer(
        self,
        bottom,
        out_channels,
        name,
        in_channels=None,
        filter_size=3,
        gate_filter_size=1,
        aux=None):
    """2D separable GRU with alexnet filters."""
    def sepgru_condition(
            step,
            timesteps,
            split_bottom,
            h,
            x_z_depth_filters,
            x_z_point_filters,
            h_z_depth_filters,
            h_z_point_filters,
            h_z_point_bias,
            x_r_depth_filters,
            x_r_point_filters,
            h_r_depth_filters,
            h_r_point_filters,
            h_r_point_bias,
            x_filter,
            h_depth_filter,
            h_point_filter,
            h_bias):
        """Condition for ending GRU."""
        return step < timesteps

    def sepgru_body(
            step,
            timesteps,
            split_bottom,
            h,
            x_z_depth_filters,
            x_z_point_filters,
            h_z_depth_filters,
            h_z_point_filters,
            h_z_point_bias,
            x_r_depth_filters,
            x_r_point_filters,
            h_r_depth_filters,
            h_r_point_filters,
            h_r_point_bias,
            x_filter,
            h_depth_filter,
            h_point_filter,
            h_bias):
        """Calculate updates for 2D separable GRU."""
        # Concatenate X_t and the hidden state
        X = tf.gather(split_bottom, step)
        if isinstance(h, list):
            h_prev = tf.gather(h, step)
        else:
            h_prev = h

        # Perform gate convolutions
        x_z = tf.nn.separable_conv2d(
            input=X,
            depthwise_filter=x_z_depth_filters,
            pointwise_filter=x_z_point_filters,
            strides=[1, 1, 1, 1],
            padding='SAME')
        x_r = tf.nn.separable_conv2d(
            input=X,
            depthwise_filter=x_r_depth_filters,
            pointwise_filter=x_r_point_filters,
            strides=[1, 1, 1, 1],
            padding='SAME')
        h_z = tf.nn.separable_conv2d(
            input=h_prev,
            depthwise_filter=h_z_depth_filters,
            pointwise_filter=h_z_point_filters,
            strides=[1, 1, 1, 1],
            padding='SAME')
        h_r = tf.nn.separable_conv2d(
            input=h_prev,
            depthwise_filter=h_r_depth_filters,
            pointwise_filter=h_r_point_filters,
            strides=[1, 1, 1, 1],
            padding='SAME')

        # Calculate gates
        z = gate_nl(x_z + h_z + h_z_point_bias)
        r = gate_nl(x_r + h_r + h_r_point_bias)

        # Update drives
        h_update = tf.squeeze(r) * h_prev

        # Perform FF/REC convolutions
        x_convs = tf.nn.conv2d(
            X,
            x_filter,
            [1, 1, 1, 1],
            padding='SAME')
        h_convs = tf.nn.separable_conv2d(
            input=h_update,
            depthwise_filter=h_depth_filter,
            pointwise_filter=h_point_filter,
            strides=[1, 1, 1, 1],
            padding='SAME')

        # Integrate circuit
        z = tf.squeeze(z)
        h_update = (z * h_prev) + ((1 - z) * cell_nl(
            x_convs + h_convs + h_bias))
        if isinstance(h, list):
            # If we are storing the hidden state at each step
            h[step] = h_update
        else:
            # If we are only keeping the final hidden state
            h = h_update
        step += 1
        return (
            step,
            timesteps,
            split_bottom,
            h,
            x_z_depth_filters,
            x_z_point_filters,
            h_z_depth_filters,
            h_z_point_filters,
            h_z_point_bias,
            x_r_depth_filters,
            x_r_point_filters,
            h_r_depth_filters,
            h_r_point_filters,
            h_r_point_bias,
            x_filter,
            h_depth_filter,
            h_point_filter,
            h_bias)

    # Scope the 2D GRU
    with tf.variable_scope(name):
        if in_channels is None:
            in_channels = int(bottom.get_shape()[-1])
        timesteps = int(bottom.get_shape()[1])

        if aux is not None and 'gate_nl' in aux.keys():
            gate_nl = aux['gate_nl']
        else:
            gate_nl = tf.sigmoid

        if aux is not None and 'cell_nl' in aux.keys():
            cell_nl = aux['cell_nl']
        else:
            cell_nl = tf.nn.relu

        if aux is not None and 'random_init' in aux.keys():
            random_init = aux['random_init']
        else:
            random_init = True

        # GRU: pack z/r/h gates into a single tensor
        # X_facing tensor, H_facing tensor for both weights and biases
        channel_multiplier = 1

        # Z
        self, x_z_depth_filters, _ = get_conv_var(
            self=self,
            filter_size=gate_filter_size,
            in_channels=in_channels,
            out_channels=channel_multiplier,
            name='%s_X_gate_depth_%s' % (name, 'z'))
        self, x_z_point_filters, _ = get_conv_var(
            self=self,
            filter_size=1,
            in_channels=in_channels * channel_multiplier,
            out_channels=out_channels,
            name='%s_X_gate_point_%s' % (name, 'z'))
        self, h_z_depth_filters, _ = get_conv_var(
            self=self,
            filter_size=gate_filter_size,
            in_channels=out_channels,  # For the hidden state
            out_channels=channel_multiplier,
            name='%s_H_gate_depth_%s' % (name, 'z'))
        self, h_z_point_filters, h_z_point_bias = get_conv_var(
            self=self,
            filter_size=1,
            in_channels=out_channels * channel_multiplier,
            out_channels=out_channels,
            name='%s_H_gate_point_%s' % (name, 'z'))

        # R
        self, x_r_depth_filters, _ = get_conv_var(
            self=self,
            filter_size=gate_filter_size,
            in_channels=in_channels,
            out_channels=channel_multiplier,
            name='%s_X_gate_depth_%s' % (name, 'r'))
        self, x_r_point_filters, _ = get_conv_var(
            self=self,
            filter_size=1,
            in_channels=in_channels * channel_multiplier,
            out_channels=out_channels,
            name='%s_X_gate_point_%s' % (name, 'r'))
        self, h_r_depth_filters, _ = get_conv_var(
            self=self,
            filter_size=gate_filter_size,
            in_channels=out_channels,  # For the hidden state
            out_channels=channel_multiplier,
            name='%s_H_gate_depth_%s' % (name, 'r'))
        self, h_r_point_filters, h_r_point_bias = get_conv_var(
            self=self,
            filter_size=1,
            in_channels=out_channels * channel_multiplier,
            out_channels=out_channels,
            name='%s_H_gate_point_%s' % (name, 'r'))

        # Split off last h weight
        self, h_depth_filter, _ = get_conv_var(
            self=self,
            filter_size=filter_size,
            in_channels=out_channels,  # For the hidden state
            out_channels=channel_multiplier,
            name='%s_H_gate_depth_%s' % (name, 'h'))
        self, h_point_filter, _ = get_conv_var(
            self=self,
            filter_size=1,
            in_channels=out_channels * channel_multiplier,
            out_channels=out_channels,
            name='%s_H_gate_point_%s' % (name, 'h'))

        # Initialize FF drive w/ alexnet filters
        assert aux is not None, 'Pass the location of alexnet weights.'
        assert 'alexnet_npy' in aux.keys(), 'Pass an alexnet_npy key.'
        alexnet_weights = np.load(aux['alexnet_npy']).item()
        alexnet_key = aux['alexnet_layer']
        alexnet_filter, alexnet_bias = alexnet_weights[alexnet_key]
        assert out_channels == alexnet_filter.shape[-1],\
            'Set weights = %s.' % alexnet_filter.shape[-1]
        if in_channels < alexnet_filter.shape[-2] and in_channels == 1:
            alexnet_filter = np.expand_dims(alexnet_filter[:, :, 0, :], axis=2)
        elif in_channels < alexnet_filter.shape[-2]:
            raise RuntimeError('Input features = %s, Alexnet features = %s' % (
                in_channels, alexnet_filter.shape[-2]))
        self, x_filter = get_var(
            self=self,
            initial_value=alexnet_filter,
            name=name,
            idx=0,
            var_name=name + "_filters")
        self, h_bias = get_var(
            self=self,
            initial_value=alexnet_bias,
            name=name,
            idx=1,
            var_name=name + "_biases")

        # Split bottom up by timesteps and initialize cell and hidden states
        split_bottom = tf.split(bottom, timesteps, axis=1)
        split_bottom = [tf.squeeze(x, axis=1) for x in split_bottom]  # Time
        h_size = [
            int(x) for x in split_bottom[0].get_shape()[:-1]] + [out_channels]
        if aux is not None and 'ff_aux' in aux.keys():
            if 'store_hidden_states' in aux['ff_aux']:
                if random_init:
                    hidden_state = [
                        tf.random_normal(
                            shape=h_size,
                            mean=0.0,
                            stddev=0.1) for x in range(timesteps)]
                else:
                    hidden_state = [
                        tf.zeros_like(split_bottom[0])
                        for x in range(timesteps)]
        else:
            if random_init:
                hidden_state = tf.random_normal(
                    shape=h_size,
                    mean=0.0,
                    stddev=0.1)
            else:
                hidden_state = tf.zeros_like(h_size)

        # While loop
        step = tf.constant(0)  # timestep iterator
        elems = [
            step,
            timesteps,
            split_bottom,
            hidden_state,
            x_z_depth_filters,
            x_z_point_filters,
            h_z_depth_filters,
            h_z_point_filters,
            h_z_point_bias,
            x_r_depth_filters,
            x_r_point_filters,
            h_r_depth_filters,
            h_r_point_filters,
            h_r_point_bias,
            x_filter,
            h_depth_filter,
            h_point_filter,
            h_bias
        ]

        if aux is not None and 'ff_aux' in aux.keys():
            if 'swap_memory' in aux['ff_aux']:
                swap_memory = aux['ff_aux']['swap_memory']
        else:
            swap_memory = True
        returned = tf.while_loop(
            sepgru_condition,
            sepgru_body,
            loop_vars=elems,
            back_prop=True,
            swap_memory=swap_memory)

        # Prepare output
        h_updated = returned[3]
        return self, h_updated


def gru2d_layer(
        self,
        bottom,
        out_channels,
        name,
        in_channels=None,
        filter_size=3,
        gate_filter_size=1,
        aux=None):
    """2D GRU convolutional layer."""
    def gru_condition(
            step,
            timesteps,
            split_bottom,
            h,
            x_gate_filters,
            h_gate_filters,
            x_filter,
            h_filter,
            gate_biases,
            h_bias):
        """Condition for ending GRU."""
        return step < timesteps

    def gru_body(
            step,
            timesteps,
            split_bottom,
            h,
            x_gate_filters,
            h_gate_filters,
            x_filter,
            h_filter,
            gate_biases,
            h_bias):
        """Calculate updates for 2D GRU."""
        # Concatenate X_t and the hidden state
        X = tf.gather(split_bottom, step)
        if isinstance(h, list):
            h_prev = tf.gather(h, step)
        else:
            h_prev = h

        # Perform gate convolutions
        x_gate_convs = tf.nn.conv2d(
            X,
            x_gate_filters,
            [1, 1, 1, 1],
            padding='SAME')
        h_gate_convs = tf.nn.conv2d(
            h_prev,
            h_gate_filters,
            [1, 1, 1, 1],
            padding='SAME')

        # Calculate gates
        gate_activities = x_gate_convs + h_gate_convs + gate_biases
        nl_activities = gate_nl(gate_activities)

        # Reshape and split into appropriate gates
        gate_sizes = [int(x) for x in nl_activities.get_shape()]
        div_g = gate_sizes[:-1] + [gate_sizes[-1] // 2, 2]
        res_gates = tf.reshape(
                nl_activities,
                div_g)
        z, r = tf.split(res_gates, 2, axis=4)

        # Update drives
        h_update = tf.squeeze(r) * h_prev

        # Perform FF/REC convolutions
        x_convs = tf.nn.conv2d(
            X,
            x_filter,
            [1, 1, 1, 1],
            padding='SAME')
        h_convs = tf.nn.conv2d(
            h_update,
            h_filter,
            [1, 1, 1, 1],
            padding='SAME')

        # Integrate circuit
        z = tf.squeeze(z)
        h_update = (z * h_prev) + ((1 - z) * cell_nl(
            x_convs + h_convs + h_bias))
        if isinstance(h, list):
            # If we are storing the hidden state at each step
            h[step] = h_update
        else:
            # If we are only keeping the final hidden state
            h = h_update
        step += 1
        return (
                step,
                timesteps,
                split_bottom,
                h,
                x_gate_filters,
                h_gate_filters,
                x_filter,
                h_filter,
                gate_biases,
                h_bias
                )

    # Scope the 2D GRU
    with tf.variable_scope(name):
        if in_channels is None:
            in_channels = int(bottom.get_shape()[-1])
        timesteps = int(bottom.get_shape()[1])

        if aux is not None and 'gate_nl' in aux.keys():
            gate_nl = aux['gate_nl']
        else:
            gate_nl = tf.sigmoid

        if aux is not None and 'cell_nl' in aux.keys():
            cell_nl = aux['cell_nl']
        else:
            cell_nl = tf.nn.relu

        if aux is not None and 'random_init' in aux.keys():
            random_init = aux['random_init']
        else:
            random_init = True

        # GRU: pack z/r/h gates into a single tensor
        # X_facing tensor, H_facing tensor for both weights and biases
        x_weights, h_weights = [], []
        biases = []
        gates = ['z', 'r']
        filter_sizes = [gate_filter_size] * 2
        for idx, (g, fs) in enumerate(zip(gates, filter_sizes)):
            self, iW, ib = get_conv_var(
                self=self,
                filter_size=fs,
                in_channels=in_channels,  # For the hidden state
                out_channels=out_channels,
                name='%s_X_gate_%s' % (name, g))
            x_weights += [iW]
            biases += [ib]
            if idx != len(gates):
                self, iW, ib = get_conv_var(
                    self=self,
                    filter_size=fs,
                    in_channels=out_channels,  # For the hidden state
                    out_channels=out_channels,
                    name='%s_H_gate_%s' % (name, g))
                h_weights += [iW]

        # Concatenate each into 3d tensors
        x_gate_filters = tf.concat(x_weights, axis=-1)
        h_gate_filters = tf.concat(h_weights, axis=-1)
        gate_biases = tf.concat(biases, axis=0)

        # Split off last h weight
        self, h_filter, h_bias = get_conv_var(
            self=self,
            filter_size=filter_size,
            in_channels=out_channels,  # For the hidden state
            out_channels=out_channels,
            name='%s_H_gate_%s' % (name, 'h'))
        self, x_filter, _ = get_conv_var(
            self=self,
            filter_size=filter_size,
            in_channels=in_channels,  # For the hidden state
            out_channels=out_channels,
            name='%s_X_gate_%s' % (name, 'x'))

        # Split bottom up by timesteps and initialize cell and hidden states
        split_bottom = tf.split(bottom, timesteps, axis=1)
        split_bottom = [tf.squeeze(x, axis=1) for x in split_bottom]  # Time
        h_size = [
            int(x) for x in split_bottom[0].get_shape()[:-1]] + [out_channels]
        if aux is not None and 'ff_aux' in aux.keys():
            if 'store_hidden_states' in aux['ff_aux']:
                if random_init:
                    hidden_state = [
                        tf.random_normal(
                            shape=h_size,
                            mean=0.0,
                            stddev=0.1) for x in range(timesteps)]
                else:
                    hidden_state = [
                        tf.zeros_like(split_bottom[0])
                        for x in range(timesteps)]
        else:
            if random_init:
                hidden_state = tf.random_normal(
                    shape=h_size,
                    mean=0.0,
                    stddev=0.1)
            else:
                hidden_state = tf.zeros_like(h_size)

        # While loop
        step = tf.constant(0)  # timestep iterator
        elems = [
            step,
            timesteps,
            split_bottom,
            hidden_state,
            x_gate_filters,
            h_gate_filters,
            x_filter,
            h_filter,
            gate_biases,
            h_bias
        ]

        if aux is not None and 'ff_aux' in aux.keys():
            if 'swap_memory' in aux['ff_aux']:
                swap_memory = aux['ff_aux']['swap_memory']
        else:
            swap_memory = True
        returned = tf.while_loop(
            gru_condition,
            gru_body,
            loop_vars=elems,
            back_prop=True,
            swap_memory=swap_memory)

        # Prepare output
        _, _, _, h_updated, _, _, _, _, _, _ = returned
        return self, h_updated


def sgru2d_layer(
        self,
        bottom,
        out_channels,
        name,
        in_channels=None,
        filter_size=3,
        gate_filter_size=1,
        aux=None):
    """2D Spatiotemporal separable GRU convolutional layer."""

    def sgru_condition(
            step,
            timesteps,
            split_bottom,
            h,
            x_gate_filters,
            h_gate_filters,
            x_filter,
            h_filter,
            gate_biases,
            h_bias):
        """Condition for ending SGRU."""
        return step < timesteps

    def sgru_body(
            step,
            timesteps,
            split_bottom,
            h,
            x_gate_filters,
            h_gate_filters,
            x_filter,
            h_filter,
            gate_biases,
            h_bias):
        """Calculate updates for 2D SGRU."""
        # Concatenate X_t and the hidden state
        X = tf.gather(split_bottom, step)
        if isinstance(h, list):
            h_prev = tf.gather(h, step)
        else:
            h_prev = h

        # Perform gate convolutions
        x_gate_convs = tf.nn.conv2d(
            X,
            x_gate_filters,
            [1, 1, 1, 1],
            padding='SAME')  # Add bias?
        h_gate_convs = tf.nn.conv2d(
            h_prev,
            h_gate_filters,
            [1, 1, 1, 1],
            padding='SAME')  # Add bias?

        # Split gates
        zx, rx = tf.split(x_gate_convs, 2, axis=4)
        zh, rh = tf.split(h_gate_convs, 2, axis=4)
        zb, rb_x, rb_h = tf.split(gate_biases, 3)  # Not sure about this
        z = tf.squeeze(gate_nl(zx + zh + zb))

        # Separately calculate input/hidden gates
        rf_a = gate_nl(rx + rb_x)  # TODO separate biases
        rh_a = gate_nl(rh + rb_h)  # TODO separate biases

        # Perform FF/REC convolutions
        x_convs = tf.nn.conv2d(
            X,
            x_filter,
            [1, 1, 1, 1],
            padding='SAME')
        h_convs = tf.nn.conv2d(
            h_prev,
            h_filter,
            [1, 1, 1, 1],
            padding='SAME')

        # Gate the FF/REC activities
        gate_x = x_convs * rf_a  # Alternatively, gate X and h_prev
        gate_h = h_convs * rh_a

        # Integrate circuit
        h_update = (z * h_prev) + ((1 - z) * cell_nl(gate_x + gate_h + h_bias))
        if isinstance(h, list):
            # If we are storing the hidden state at each step
            h[step] = h_update
        else:
            # If we are only keeping the final hidden state
            h = h_update
        step += 1
        return (
                step,
                timesteps,
                split_bottom,
                h,
                x_gate_filters,
                h_gate_filters,
                x_filter,
                h_filter,
                gate_biases,
                h_bias
                )

    # Scope the 2D SGRU
    with tf.variable_scope(name):
        if in_channels is None:
            # Channels for the input x
            in_channels = int(bottom.get_shape()[-1])
        timesteps = int(bottom.get_shape()[1])

        if aux is not None and 'gate_nl' in aux.keys():
            gate_nl = aux['gate_nl']
        else:
            gate_nl = tf.sigmoid  # @Michele, try hard sigmpoid

        if aux is not None and 'cell_nl' in aux.keys():
            cell_nl = aux['cell_nl']
        else:
            cell_nl = tf.nn.relu  # @Michele, try relu

        if aux is not None and 'random_init' in aux.keys():
            random_init = aux['random_init']
        else:
            random_init = True

        # GRU: pack z/r/h gates into a single tensor
        # X_facing tensor, H_facing tensor for both weights and biases
        x_weights, h_weights = [], []
        biases = []
        gates = ['z', 'r']
        filter_sizes = [gate_filter_size]
        for idx, (g, fs) in enumerate(zip(gates, filter_sizes)):
            self, iW, ib = get_conv_var(
                self=self,
                filter_size=fs,
                in_channels=in_channels,  # For the hidden state
                out_channels=out_channels,
                name='%s_X_gate_%s' % (name, g))
            x_weights += [iW]
            biases += [ib]
            if idx != len(gates):
                self, iW, ib = get_conv_var(
                    self=self,
                    filter_size=fs,
                    in_channels=out_channels,  # For the hidden state
                    out_channels=out_channels,
                    name='%s_H_gate_%s' % (name, g))
                h_weights += [iW]

        # Concatenate each into 3d tensors
        x_gate_filters = tf.concat(x_weights, axis=-1)
        h_gate_filters = tf.concat(h_weights, axis=-1)
        gate_biases = tf.concat(biases, axis=0)

        # Create weights for H and X (U/W)
        self, h_filter, h_bias = get_conv_var(
            self=self,
            filter_size=filter_size,
            in_channels=out_channels,  # For the hidden state
            out_channels=out_channels,
            name='%s_H_gate_%s' % (name, 'h'))
        self, x_filter, _ = get_conv_var(
            self=self,
            filter_size=filter_size,
            in_channels=in_channels,  # For the hidden state
            out_channels=out_channels,
            name='%s_X_gate_%s' % (name, 'x'))

        # Split bottom up by timesteps and initialize cell and hidden states
        split_bottom = tf.split(bottom, timesteps, axis=1)
        split_bottom = [tf.squeeze(x, axis=1) for x in split_bottom]  # Time
        h_size = [
            int(x) for x in split_bottom[0].get_shape()[:-1]] + [out_channels]
        if aux is not None and 'ff_aux' in aux.keys():
            if 'store_hidden_states' in aux['ff_aux']:
                if random_init:
                    hidden_state = [
                        tf.random_normal(
                            shape=h_size,
                            mean=0.0,
                            stddev=0.1) for x in range(timesteps)]
                else:
                    hidden_state = [
                        tf.zeros_like(split_bottom[0])
                        for x in range(timesteps)]
        else:
            if random_init:
                hidden_state = tf.random_normal(
                    shape=h_size,
                    mean=0.0,
                    stddev=0.1)
            else:
                hidden_state = tf.zeros_like(h_size)

        # While loop
        step = tf.constant(0)  # timestep iterator
        elems = [
            step,
            timesteps,
            split_bottom,
            hidden_state,
            x_gate_filters,
            h_gate_filters,
            x_filter,
            h_filter,
            gate_biases,
            h_bias
        ]

        if aux is not None and 'ff_aux' in aux.keys():
            if 'swap_memory' in aux['ff_aux']:
                swap_memory = aux['ff_aux']['swap_memory']
        else:
            swap_memory = True
        returned = tf.while_loop(
            sgru_condition,
            sgru_body,
            loop_vars=elems,
            back_prop=True,
            swap_memory=swap_memory)

        # Prepare output
        _, _, _, h_updated, _, _, _, _, _, _ = returned
        return self, h_updated


def mru2d_layer(
        self,
        bottom,
        out_channels,
        name,
        in_channels=None,
        filter_size=3,
        gate_filter_size=1,
        aux=None):
    """2D MRU convolutional layer."""

    def mru_condition(
            step,
            timesteps,
            split_bottom,
            h,
            x_gate_filters,
            h_gate_filters,
            x_filter,
            h_filter,
            gate_biases,
            h_bias):
        """Condition for ending MRU."""
        return step < timesteps

    def mru_body(
            step,
            timesteps,
            split_bottom,
            h,
            x_gate_filters,
            h_gate_filters,
            x_filter,
            h_filter,
            gate_biases,
            h_bias):
        """Calculate updates for 2D MRU."""
        # Concatenate X_t and the hidden state
        X = tf.gather(split_bottom, step)
        if isinstance(h, list):
            h_prev = tf.gather(h, step)
        else:
            h_prev = h

        # Perform gate convolutions
        x_gate_convs = tf.nn.conv2d(
            X,
            x_gate_filters,
            [1, 1, 1, 1],
            padding='SAME')
        h_gate_convs = tf.nn.conv2d(
            h_prev,
            h_gate_filters,
            [1, 1, 1, 1],
            padding='SAME')

        # Calculate gates
        gate_activities = x_gate_convs + h_gate_convs + gate_biases
        z = gate_nl(gate_activities)

        # Perform FF/REC convolutions
        x_convs = tf.nn.conv2d(
            X,
            x_filter,
            [1, 1, 1, 1],
            padding='SAME')
        h_convs = tf.nn.conv2d(
            h_prev,
            h_filter,
            [1, 1, 1, 1],
            padding='SAME')

        # Integrate circuit
        z = tf.squeeze(z)
        h_update = (z * h_prev) + (
            (1 - z) * cell_nl(x_convs + h_convs + h_bias))
        if isinstance(h, list):
            # If we are storing the hidden state at each step
            h[step] = h_update
        else:
            # If we are only keeping the final hidden state
            h = h_update
        step += 1
        return (
                step,
                timesteps,
                split_bottom,
                h,
                x_gate_filters,
                h_gate_filters,
                x_filter,
                h_filter,
                gate_biases,
                h_bias
                )

    # Scope the 2D MRU
    with tf.variable_scope(name):
        if in_channels is None:
            in_channels = int(bottom.get_shape()[-1])
        timesteps = int(bottom.get_shape()[1])

        if aux is not None and 'gate_nl' in aux.keys():
            gate_nl = aux['gate_nl']
        else:
            gate_nl = tf.sigmoid

        if aux is not None and 'cell_nl' in aux.keys():
            cell_nl = aux['cell_nl']
        else:
            cell_nl = tf.nn.relu

        if aux is not None and 'random_init' in aux.keys():
            random_init = aux['random_init']
        else:
            random_init = True

        # GRU: pack z/r/h gates into a single tensor
        # X_facing tensor, H_facing tensor for both weights and biases
        x_weights, h_weights = [], []
        biases = []
        gates = ['z']
        filter_sizes = [gate_filter_size]
        for idx, (g, fs) in enumerate(zip(gates, filter_sizes)):
            _, iW, ib = get_conv_var(
                self=self,
                filter_size=fs,
                in_channels=in_channels,  # For the hidden state
                out_channels=out_channels,
                name='%s_X_gate_%s' % (name, g))
            x_weights += [iW]
            biases += [ib]
            if idx != len(gates):
                _, iW, ib = get_conv_var(
                    self=self,
                    filter_size=fs,
                    in_channels=out_channels,  # For the hidden state
                    out_channels=out_channels,
                    name='%s_H_gate_%s' % (name, g))
                h_weights += [iW]

        # Concatenate each into 3d tensors
        x_gate_filters = tf.concat(x_weights, axis=-1)
        h_gate_filters = tf.concat(h_weights, axis=-1)
        gate_biases = tf.concat(biases, axis=0)

        # Split off last h weight
        self, h_filter, h_bias = get_conv_var(
            self=self,
            filter_size=filter_size,
            in_channels=out_channels,  # For the hidden state
            out_channels=out_channels,
            name='%s_H_gate_%s' % (name, 'h'))
        self, x_filter, _ = get_conv_var(
            self=self,
            filter_size=filter_size,
            in_channels=in_channels,  # For the hidden state
            out_channels=out_channels,
            name='%s_X_gate_%s' % (name, 'x'))

        # Reshape bottom so that timesteps are first
        res_bottom = tf.reshape(
            bottom,
            np.asarray([int(x) for x in bottom.get_shape()])[[1, 0, 2, 3, 4]])
        h_size = [
            int(x) for x in res_bottom.get_shape()][1: -1] + [out_channels]
        if aux is not None and 'ff_aux' in aux.keys():
            if 'store_hidden_states' in aux['ff_aux']:
                # Store all hidden states in a list
                if random_init:
                    hidden_state = [
                        tf.random_normal(
                            shape=h_size,
                            mean=0.0,
                            stddev=0.1) for x in range(timesteps)]
                else:
                    hidden_state = [
                        tf.zeros(h_size, dtype=tf.float32)
                        for x in range(timesteps)]
        else:
            if random_init:
                hidden_state = tf.random_normal(
                    shape=h_size,
                    mean=0.0,
                    stddev=0.1)
            else:
                hidden_state = tf.zeros(h_size, dtype=tf.float32)

        # While loop
        step = tf.constant(0)  # timestep iterator
        elems = [
            step,
            timesteps,
            res_bottom,
            hidden_state,
            x_gate_filters,
            h_gate_filters,
            x_filter,
            h_filter,
            gate_biases,
            h_bias
        ]

        if aux is not None and 'ff_aux' in aux.keys():
            if 'swap_memory' in aux['ff_aux']:
                swap_memory = aux['ff_aux']['swap_memory']
        else:
            swap_memory = False
        returned = tf.while_loop(
            mru_condition,
            mru_body,
            loop_vars=elems,
            back_prop=True,
            swap_memory=swap_memory)

        # Prepare output
        _, _, _, h_updated, _, _, _, _, _, _ = returned
        return self, h_updated


def rnn2d_layer(
        self,
        bottom,
        out_channels,
        name,
        in_channels=None,
        filter_size=3,
        gate_filter_size=1,
        aux=None):
    """2D RNN convolutional layer."""

    def rnn_condition(
            step,
            timesteps,
            split_bottom,
            h,
            x_filter,
            h_filter,
            h_bias):
        """Condition for ending MRU."""
        return step < timesteps

    def rnn_body(
            step,
            timesteps,
            split_bottom,
            h,
            x_filter,
            h_filter,
            h_bias):
        """Calculate updates for 2D MRU."""
        # Concatenate X_t and the hidden state
        X = tf.gather(split_bottom, step)
        if isinstance(h, list):
            h_prev = tf.gather(h, step)
        else:
            h_prev = h

        # Perform FF/REC convolutions
        x_convs = tf.nn.conv2d(
            X,
            x_filter,
            [1, 1, 1, 1],
            padding='SAME')
        h_convs = tf.nn.conv2d(
            h_prev,
            h_filter,
            [1, 1, 1, 1],
            padding='SAME')

        # Integrate circuit
        h_update = cell_nl(x_convs + h_convs + h_bias)
        if isinstance(h, list):
            # If we are storing the hidden state at each step
            h[step] = h_update
        else:
            # If we are only keeping the final hidden state
            h = h_update
        step += 1
        return (
                step,
                timesteps,
                split_bottom,
                h,
                x_filter,
                h_filter,
                h_bias
                )

    # Scope the 2D RNN
    with tf.variable_scope(name):
        if in_channels is None:
            in_channels = int(bottom.get_shape()[-1])
        timesteps = int(bottom.get_shape()[1])

        if aux is not None and 'cell_nl' in aux.keys():
            cell_nl = aux['cell_nl']
        else:
            cell_nl = activations()['selu']

        if aux is not None and 'random_init' in aux.keys():
            random_init = aux['random_init']
        else:
            random_init = True

        # Only X/H weights
        self, h_filter, h_bias = get_conv_var(
            self=self,
            filter_size=filter_size,
            in_channels=out_channels,  # For the hidden state
            out_channels=out_channels,
            # init_type='identity',  # https://arxiv.org/abs/1504.00941
            name='%s_H_gate_%s' % (name, 'h'))
        self, x_filter, _ = get_conv_var(
            self=self,
            filter_size=filter_size,
            in_channels=in_channels,  # For the hidden state
            out_channels=out_channels,
            name='%s_X_gate_%s' % (name, 'x'))

        # Reshape bottom so that timesteps are first
        res_bottom = tf.reshape(
            bottom,
            np.asarray([int(x) for x in bottom.get_shape()])[[1, 0, 2, 3, 4]])
        h_size = [
            int(x) for x in res_bottom.get_shape()][1: -1] + [out_channels]
        if aux is not None and 'ff_aux' in aux.keys():
            if 'store_hidden_states' in aux['ff_aux']:
                # Store all hidden states in a list
                if random_init:
                    hidden_state = [
                        tf.random_normal(
                            shape=h_size,
                            mean=0.0,
                            stddev=0.1) for x in range(timesteps)]
                else:
                    hidden_state = [
                        tf.zeros(h_size, dtype=tf.float32)
                        for x in range(timesteps)]
        else:
            if random_init:
                hidden_state = tf.random_normal(
                    shape=h_size,
                    mean=0.0,
                    stddev=0.1)
            else:
                hidden_state = tf.zeros(h_size, dtype=tf.float32)

        # While loop
        step = tf.constant(0)  # timestep iterator
        elems = [
            step,
            timesteps,
            res_bottom,
            hidden_state,
            x_filter,
            h_filter,
            h_bias
        ]

        if aux is not None and 'ff_aux' in aux.keys():
            if 'swap_memory' in aux['ff_aux']:
                swap_memory = aux['ff_aux']['swap_memory']
        else:
            swap_memory = False
        returned = tf.while_loop(
            rnn_condition,
            rnn_body,
            loop_vars=elems,
            back_prop=True,
            swap_memory=swap_memory)

        # Prepare output
        _, _, _, h_updated, _, _, _ = returned
        return self, hidden_state


def conv3d_layer(
        self,
        bottom,
        out_channels,
        name,
        in_channels=None,
        filter_size=3,
        stride=[1, 1, 1, 1, 1],
        padding='SAME',
        aux=None):
    """3D convolutional layer."""
    with tf.variable_scope(name):
        if in_channels is None:
            in_channels = int(bottom.get_shape()[-1])
        timesteps = int(bottom.get_shape()[1])
        kernel = [timesteps, filter_size, filter_size]
        self, filt, conv_biases = get_conv_var(
            self=self,
            filter_size=filter_size,
            in_channels=in_channels,
            out_channels=out_channels,
            name=name,
            kernel=kernel)
        conv = tf.nn.conv3d(
            bottom,
            filt,
            stride,
            padding=padding)
        bias = tf.nn.bias_add(conv, conv_biases)
        return self, bias


def st_resnet_layer(
        self,
        bottom,
        layer_weights,
        name,
        aux=None,
        combination=None):  # tf.multiply
    """Spatiotemporal residual layer."""
    ln = '%s_branch' % name
    rlayer = tf.identity(bottom)
    if aux is not None:
        if 'activation' in aux.keys():
            activation = aux['activation']
        if 'normalization' in aux.keys():
            normalization = aux['normalization']
        if 'ff_aux' in aux.keys():
            if aux['ff_aux']['combination'] == 'add':
                combination = tf.add
            elif aux['ff_aux']['combination'] == 'prod':
                combination = tf.multiply
            elif aux['ff_aux']['combination'] == 'add_prod':
                combination = lambda x, y: tf.concat(
                        tf.add(x, y),
                        tf.multiply(x, y)
                    )
        else:
            combination = tf.add
    if normalization is not None:
        if normalization is not 'batch':
            raise RuntimeError(
                'Normalization not yet implemented for non-batchnorm.')
        nm = normalizations({'training': self.training})[normalization]
    else:
        nm = lambda x: x
    if activation is not None:
        ac = activations()[activation]
    else:
        ac = lambda x: x
    for idx, lw in enumerate(layer_weights):
        ln = '%s_%s' % (name, idx)
        self, rlayer = conv3d_layer(
            self=self,
            bottom=rlayer,
            in_channels=int(rlayer.get_shape()[-1]),
            out_channels=lw,
            name=ln)
        rlayer = nm(ac(rlayer), None, None, None)
        if isinstance(rlayer, tuple):
            rlayer = rlayer[0]
    return self, combination(rlayer, bottom)


def fc_layer(self, bottom, out_channels, name, in_channels=None):
    """Fully connected layer."""
    with tf.variable_scope(name):
        if in_channels is None:
            in_channels = int(bottom.get_shape()[-1])
        self, weights, biases = get_fc_var(
            self=self,
            in_size=in_channels,
            out_size=out_channels,
            name=name)

        x = tf.reshape(bottom, [-1, in_channels])
        fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

        return self, fc


def multi_fc_layer(self, bottom, out_channels, name, in_channels=None):
    """Fully connected layer."""
    with tf.variable_scope(name):
        if in_channels is None:
            in_channels = int(bottom.get_shape()[-1])
        fcs = []
        for idx, oc in enumerate(out_channels):
            self, weights, biases = get_fc_var(
                self=self,
                in_size=in_channels,
                out_size=oc,
                name='%s_%s' % (name, idx))

            x = tf.reshape(bottom, [-1, in_channels])
            fcs += [tf.nn.bias_add(tf.matmul(x, weights), biases)]
        return self, fcs


def sparse_pool_layer(
        self,
        bottom,
        out_channels,
        name,
        in_channels=None,
        aux=None):
    """Sparse pooling layer."""
    def create_gaussian_rf(xy, h, w):
        """Create a gaussian bump for initializing the spatial weights."""
        # TODO: implement this.
        raise NotImplementedError

    assert len(bottom.get_shape()) > 2, \
        'Sparse pooling requires a tensor input.'
    with tf.variable_scope(name):
        bottom_shape = [int(x) for x in bottom.get_shape()]

        # HxW spatial weights
        spatial_weights = tf.get_variable(
            name='%s_spatial' % name,
            dtype=tf.float32,
            initializer=initialization.xavier_initializer(
                shape=[1, bottom_shape[1], bottom_shape[2], 1],
                mask=None))
        # If supplied, initialize the spatial weights with RF info
        if aux is not None and 'xy' in aux.keys():
            gaussian_xy = aux['xy']
            if 'h' in aux.keys():
                gaussian_h = aux['h']
                gaussian_w = aux['w']
                k = aux['k']
            else:
                gaussian_h, gaussian_w, k = None, None, None
            spatial_rf = create_gaussian_rf(
                xy=gaussian_xy,
                h=gaussian_h,
                w=gaussian_w,
                k=k)
            spatial_weights += spatial_rf
        spatial_sparse = tf.reduce_mean(
            bottom * spatial_weights, reduction_indices=[1, 2])

        # K channel weights
        spatial_shape = spatial_sparse.get_shape().as_list()
        channel_weights = tf.get_variable(
            name='%s_channel' % name,
            dtype=tf.float32,
            initializer=initialization.xavier_initializer(
                shape=[spatial_shape[-1], 1],
                uniform=True,
                mask=None))
        output = tf.squeeze(
            tf.matmul(spatial_sparse, channel_weights), axis=-1)
        return self, output


def get_conv_var(
        self,
        filter_size,
        in_channels,
        out_channels,
        name,
        init_type='xavier',
        kernel=None):
    """Prepare convolutional kernel weights."""
    if kernel is None:
        kernel = [filter_size] * 2
    if init_type == 'xavier':
        weight_init = [
            kernel + [in_channels, out_channels],
            tf.contrib.layers.xavier_initializer_conv2d(uniform=False)]
    elif init_type == 'identity':
        raise NotImplementedError  # TODO: Update TF and fix this
        weight_init = [
            kernel + [in_channels, out_channels],
            initialization.Identity()]
    else:
        weight_init = tf.truncated_normal(
            [filter_size, filter_size, in_channels, out_channels],
            0.0, 0.001)
    bias_init = tf.truncated_normal([out_channels], .0, .001)
    self, filters = get_var(
        self=self,
        initial_value=weight_init,
        name=name,
        idx=0,
        var_name=name + "_filters")
    self, biases = get_var(
        self=self,
        initial_value=bias_init,
        name=name,
        idx=1,
        var_name=name + "_biases")
    return self, filters, biases


def get_fc_var(
        self,
        in_size,
        out_size,
        name,
        init_type='xavier'):
    """Prepare fully connected weights."""
    if init_type == 'xavier':
        weight_init = [
            [in_size, out_size],
            tf.contrib.layers.xavier_initializer(uniform=False)]
    else:
        weight_init = tf.truncated_normal(
            [in_size, out_size], 0.0, 0.001)
    bias_init = tf.truncated_normal([out_size], .0, .001)
    self, weights = get_var(
        self=self,
        initial_value=weight_init,
        name=name,
        idx=0,
        var_name=name + "_weights")
    self, biases = get_var(
        self=self,
        initial_value=bias_init,
        name=name,
        idx=1,
        var_name=name + "_biases")
    return self, weights, biases


def get_var(
        self,
        initial_value,
        name,
        idx,
        var_name,
        in_size=None,
        out_size=None):
    """Handle variable loading if necessary."""
    if self.data_dict is not None and name in self.data_dict:
        value = self.data_dict[name][idx]
    else:
        value = initial_value

    # get_variable, change the boolian to numpy
    if type(value) is list:
        var = tf.get_variable(
            name=var_name,
            shape=value[0],
            initializer=value[1],
            trainable=self.training)
    else:
        var = tf.get_variable(
            name=var_name,
            initializer=value,
            trainable=self.training)
    self.var_dict[(name, idx)] = var
    return self, var


def pool_ff_interpreter(
        self,
        it_neuron_op,
        act,
        it_name,
        it_dict):
    """Wrapper for FF and pooling functions. DEPRECIATED."""
    if it_neuron_op == 'pool':  # TODO create wrapper for FF ops.
        self, act = pool.max_pool(
            self=self,
            bottom=act,
            name=it_name
        )
    elif it_neuron_op == 'pool3d':  # TODO create wrapper for FF ops.
        self, act = pool.max_pool_3d(
            self=self,
            bottom=act,
            name=it_name
        )
    elif it_neuron_op == 'dog' or it_neuron_op == 'DoG':
        self, act = dog_layer(
            self=self,
            bottom=act,
            layer_weights=it_dict['weights'],
            name=it_name,
        )
    elif it_neuron_op == 'conv':
        self, act = conv_layer(
            self=self,
            bottom=act,
            in_channels=int(act.get_shape()[-1]),
            out_channels=it_dict['weights'][0],
            name=it_name,
            filter_size=it_dict['filter_size'][0]
        )
    elif it_neuron_op == 'conv3d':
        self, act = conv3d_layer(
            self=self,
            bottom=act,
            in_channels=int(act.get_shape()[-1]),
            out_channels=it_dict['weights'][0],
            name=it_name,
            filter_size=it_dict['filter_size'][0],
            aux=it_dict['aux']
        )
    elif it_neuron_op == 'residual_conv3d':
        pass
    elif it_neuron_op == 'fc':
        self, act = fc_layer(
            self=self,
            bottom=act,
            in_channels=int(act.get_shape()[-1]),
            out_channels=it_dict['weights'][0],
            name=it_name)
    elif it_neuron_op == 'sparse_pool':
        self, act = sparse_pool_layer(
            self=self,
            bottom=act,
            in_channels=int(act.get_shape()[-1]),
            out_channels=it_dict['weights'][0],
            aux=it_dict,
            name=it_name)
    elif it_neuron_op == 'res':
        self, act = resnet_layer(
            self=self,
            bottom=act,
            aux=it_dict['aux'],
            layer_weights=it_dict['weights'],
            name=it_name)
    elif it_neuron_op == 'gather':
        self, act = gather_value_layer(
            self=self,
            bottom=act,
            aux=it_dict['aux'],
            name=it_name)
    elif it_neuron_op == 'pass':
        pass
    else:
        raise RuntimeError(
            'Your specified operation %s is not implemented' % it_neuron_op)
    return self, act
