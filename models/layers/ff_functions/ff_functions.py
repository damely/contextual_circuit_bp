"""Functions for feedforward models."""
import numpy as np
import tensorflow as tf
from ops import initialization
from models.layers.activations import activations
from models.layers.normalizations import normalizations
# from models.layers import pool


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


def hermann_gather_value_layer(
        self,
        bottom,
        aux,
        name,
        eps=1e-12):
    """Gather a value from a location in an activity tensor."""
    assert aux is not None,\
        'Gather op needs an aux dict with h/w coordinates.'
    assert 'h' in aux.keys() and 'w' in aux.keys(),\
        'Gather op dict needs h/w key value pairs'
    # Take the mean of bottom across the first dimension (N)
    mean_bottom = tf.reduce_mean(bottom, reduction_indices=[0], keepdims=True)
    h_gap = aux['h_gap']
    w_gap = aux['w_gap']
    h_intersection = aux['h_intersection']
    w_intersection = aux['w_intersection']
    gap_score = tf.squeeze(
        mean_bottom[:, h_gap, w_gap, :]) ** 2
    intersection_score = tf.squeeze(
        mean_bottom[:, h_intersection, w_intersection, :]) ** 2
    out = gap_score / (intersection_score + eps)
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
    def combination_fun(x, y):
        return tf.concat(
            tf.add(x, y),
            tf.multiply(x, y))

    def pass_x(x):
        return x
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
                combination = combination_fun
        else:
            combination = tf.add
    if normalization is not None:
        if normalization is not 'batch':
            raise RuntimeError(
                'Normalization not yet implemented for non-batchnorm.')
        nm = normalizations({'training': self.training})[normalization]
    else:
        nm = pass_x
    if activation is not None:
        ac = activations()[activation]
    else:
        ac = pass_x
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
    if 'stride' in aux.keys():
        stride = aux['stride']
    train_alexnet, init_bias = True, False
    if 'trainable' in aux.keys():
        train_alexnet = aux['trainable']
    if 'init_bias' in aux.keys():
        init_bias = aux['init_bias']
    if 'rescale' in aux.keys():
        rescale = aux['rescale']
    else:
        rescale = False
    alexnet_weights = np.load(aux['alexnet_npy']).item()
    alexnet_key = aux['alexnet_layer']
    alexnet_filter, alexnet_bias = alexnet_weights[alexnet_key]
    with tf.variable_scope(name):
        if in_channels is None:
            in_channels = int(bottom.get_shape()[-1])
        if out_channels != alexnet_filter.shape[-1]:
            out_channels = alexnet_filter.shape[-1]
            print 'Set weights = %s.' % alexnet_filter.shape[-1]
        if in_channels < alexnet_filter.shape[-2] and in_channels == 1:
            alexnet_filter = np.mean(alexnet_filter, axis=2, keepdims=True)
        elif in_channels < alexnet_filter.shape[-2]:
            raise RuntimeError('Input features = %s, Alexnet features = %s' % (
                in_channels, alexnet_filter.shape[-2]))
        filters = tf.get_variable(
            name=name + "_filters",
            initializer=alexnet_filter,
            trainable=train_alexnet)
        if rescale:
            rescaler = tf.get_variable(
                name=name + "_scale",
                initializer=tf.truncated_normal([out_channels], .0, .001),
                trainable=True)
            filters = filters * rescaler
        self.var_dict[(name, 0)] = filters
        if init_bias or len(alexnet_bias) == 0:
            alexnet_bias = tf.truncated_normal([out_channels], .0, .001)
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
            in_channels=in_channels * multiplier,
            out_channels=out_channels,
            name='p_%s' % name)
        conv = tf.nn.separable_conv2d(
            input=bottom,
            depthwise_filter=dfilt,
            pointwise_filter=pfilt,
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


def dog_3d_layer(
        self,
        bottom,
        out_channels,
        name,
        in_channels=None,
        filter_size=3,
        stride=[1, 1, 1, 1, 1],
        padding='SAME',
        aux=None):
    """DoG concatenated across time."""
    batch_size, timesteps = int(bottom.get_shape()[:2])
    activities = []
    for idx in range(timesteps):
        activities += [dog_layer(
            self,
            tf.squeeze(bottom[:, idx, :, :, :]),
            layer_weights,
            '%s_%s' % (name, idx),
            init_weight=10.,
            model_dtype=tf.float32)]
    reshaped_activities = tf.reshape(
        tf.transpose(
            tf.stack(activities), [1, 0, 2]),
        [batch_size, -1])
    return reshaped_activities


def st_resnet_layer(
        self,
        bottom,
        layer_weights,
        name,
        aux=None,
        combination=None):  # tf.multiply
    """Spatiotemporal residual layer."""
    def combination_fun(x, y):
        return tf.concat(
            tf.add(x, y),
            tf.multiply(x, y))

    def pass_x(x):
        return x
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
                combination = combination_fun
        else:
            combination = tf.add
    if normalization is not None:
        if normalization is not 'batch':
            raise RuntimeError(
                'Normalization not yet implemented for non-batchnorm.')
        nm = normalizations({'training': self.training})[normalization]
    else:
        nm = pass_x
    if activation is not None:
        ac = activations()[activation]
    else:
        ac = pass_x
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
