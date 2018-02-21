"""Functions for handling feedforward and pooling operations."""
# import numpy as np
# import tensorflow as tf
# from ops import initialization
# from models.layers.activations import activations
# from models.layers.normalizations import normalizations
from models.layers.ff_functions import ff_functions as ff_fun
from models.layers.ff_functions import recurrent_functions as rf_fun
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
        context, act = ff_fun.gather_value_layer(
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
        context, act = ff_fun.bias_layer(
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
        context, act = ff_fun.dog_layer(
            self=context,
            bottom=act,
            layer_weights=out_channels,
            name=name)
        return context, act

    def dog_3d(
            self,
            context,
            act,
            in_channels,
            out_channels,
            filter_size,
            name,
            it_dict):
        """Add a temporal Difference of Gaussians layer."""
        context, act = ff_fun.dog_3d_layer(
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
        context, act = ff_fun.dog_layer(
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
        context, act = ff_fun.dog_conv_layer(
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
        context, act = ff_fun.gabor_conv_layer(
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
        context, act = ff_fun.conv_layer(
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
        context, act = ff_fun.alexnet_conv_layer(
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
        context, act = ff_fun.conv3d_layer(
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
        context, act = ff_fun.sep_conv_layer(
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
        context, act = ff_fun.time_sep_conv3d_layer(
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
        context, act = ff_fun.complete_sep_conv3d_layer(
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
        context, act = rf_fun.lstm2d_layer(
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
        context, act = rf_fun.sgru2d_layer(
            self=context,
            bottom=act,
            in_channels=in_channels,
            out_channels=out_channels,
            name=name,
            filter_size=filter_size,
            aux=it_dict)
        return context, act

    def alexnet_sgru2d(
            self,
            context,
            act,
            in_channels,
            out_channels,
            filter_size,
            name,
            it_dict):
        """Convolutional separable GRU."""
        context, act = rf_fun.alexnet_sgru2d_layer(
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
        context, act = rf_fun.gru2d_layer(
            self=context,
            bottom=act,
            in_channels=in_channels,
            out_channels=out_channels,
            name=name,
            filter_size=filter_size,
            aux=it_dict)
        return context, act

    def alexnet_gru2d(
            self,
            context,
            act,
            in_channels,
            out_channels,
            filter_size,
            name,
            it_dict):
        """Convolutional GRU."""
        context, act = rf_fun.alexnet_gru2d_layer(
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
        context, act = rf_fun.sepgru2d_layer(
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
        context, act = rf_fun.alexnet_sepgru2d_layer(
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
        context, act = rf_fun.mru2d_layer(
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
        context, act = rf_fun.rnn2d_layer(
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
        context, act = rf_fun.lstm1d_layer(
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
        context, act = rf_fun.gru1d_layer(
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
        context, act = ff_fun.conv1d_layer(
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
        context, act = ff_fun.fc_layer(
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
        context, act = ff_fun.multi_fc_layer(
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
        context, act = ff_fun.sparse_pool_layer(
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
        context, act = ff_fun.resnet_layer(
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
