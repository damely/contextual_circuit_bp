"""Functions for recurrent models."""
import numpy as np
import tensorflow as tf
# from ops import initialization
from models.layers.activations import activations
# from models.layers.normalizations import normalizations
# from models.layers import pool
from models.layers.ff_functions import ff_functions as ff


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
            self, iW, ib = ff.get_fc_var(
                self=self,
                in_size=in_channels,
                out_size=out_channels,
                name='%s_X_gate_%s' % (name, g))
            x_weights += [iW]
            biases += [ib]
            self, iW, ib = ff.get_fc_var(
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
            self, iW, ib = ff.get_fc_var(
                self=self,
                in_size=in_channels,
                out_size=out_channels,
                name='%s_X_gate_%s' % (name, g))
            x_weights += [iW]
            biases += [ib]
            self, iW, _ = ff.get_fc_var(
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
        self, x_filter, _ = ff.get_fc_var(
            self=self,
            in_size=in_channels,
            out_size=out_channels,
            name='%s_X_gate_%s' % (name, 'x'))
        self, h_filter, h_bias = ff.get_fc_var(
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
            self, iW, ib = ff.get_conv_var(
                self=self,
                filter_size=fs,
                in_channels=in_channels,  # For the hidden state
                out_channels=out_channels,
                name='%s_X_gate_%s' % (name, g))
            x_weights += [iW]
            biases += [ib]
            self, iW, ib = ff.get_conv_var(
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
        self, x_z_depth_filters, _ = ff.get_conv_var(
            self=self,
            filter_size=gate_filter_size,
            in_channels=in_channels,
            out_channels=channel_multiplier,
            name='%s_X_gate_depth_%s' % (name, 'z'))
        self, x_z_point_filters, _ = ff.get_conv_var(
            self=self,
            filter_size=1,
            in_channels=in_channels * channel_multiplier,
            out_channels=out_channels,
            name='%s_X_gate_point_%s' % (name, 'z'))
        self, h_z_depth_filters, _ = ff.get_conv_var(
            self=self,
            filter_size=gate_filter_size,
            in_channels=out_channels,  # For the hidden state
            out_channels=channel_multiplier,
            name='%s_H_gate_depth_%s' % (name, 'z'))
        self, h_z_point_filters, h_z_point_bias = ff.get_conv_var(
            self=self,
            filter_size=1,
            in_channels=out_channels * channel_multiplier,
            out_channels=out_channels,
            name='%s_H_gate_point_%s' % (name, 'z'))

        # R
        self, x_r_depth_filters, _ = ff.get_conv_var(
            self=self,
            filter_size=gate_filter_size,
            in_channels=in_channels,
            out_channels=channel_multiplier,
            name='%s_X_gate_depth_%s' % (name, 'r'))
        self, x_r_point_filters, _ = ff.get_conv_var(
            self=self,
            filter_size=1,
            in_channels=in_channels * channel_multiplier,
            out_channels=out_channels,
            name='%s_X_gate_point_%s' % (name, 'r'))
        self, h_r_depth_filters, _ = ff.get_conv_var(
            self=self,
            filter_size=gate_filter_size,
            in_channels=out_channels,  # For the hidden state
            out_channels=channel_multiplier,
            name='%s_H_gate_depth_%s' % (name, 'r'))
        self, h_r_point_filters, h_r_point_bias = ff.get_conv_var(
            self=self,
            filter_size=1,
            in_channels=out_channels * channel_multiplier,
            out_channels=out_channels,
            name='%s_H_gate_point_%s' % (name, 'r'))

        # Split off last h weight
        self, h_depth_filter, h_bias = ff.get_conv_var(
            self=self,
            filter_size=filter_size,
            in_channels=out_channels,  # For the hidden state
            out_channels=channel_multiplier,
            name='%s_H_gate_depth_%s' % (name, 'h'))
        self, h_point_filter, h_bias = ff.get_conv_var(
            self=self,
            filter_size=1,
            in_channels=out_channels * channel_multiplier,
            out_channels=out_channels,
            name='%s_H_gate_point_%s' % (name, 'h'))
        self, x_depth_filter, _ = ff.get_conv_var(
            self=self,
            filter_size=filter_size,
            in_channels=in_channels,  # For the hidden state
            out_channels=channel_multiplier,
            name='%s_X_gate_depth_%s' % (name, 'x'))
        self, x_point_filter, _ = ff.get_conv_var(
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
        self, x_z_depth_filters, _ = ff.get_conv_var(
            self=self,
            filter_size=gate_filter_size,
            in_channels=in_channels,
            out_channels=channel_multiplier,
            name='%s_X_gate_depth_%s' % (name, 'z'))
        self, x_z_point_filters, _ = ff.get_conv_var(
            self=self,
            filter_size=1,
            in_channels=in_channels * channel_multiplier,
            out_channels=out_channels,
            name='%s_X_gate_point_%s' % (name, 'z'))
        self, h_z_depth_filters, _ = ff.get_conv_var(
            self=self,
            filter_size=gate_filter_size,
            in_channels=out_channels,  # For the hidden state
            out_channels=channel_multiplier,
            name='%s_H_gate_depth_%s' % (name, 'z'))
        self, h_z_point_filters, h_z_point_bias = ff.get_conv_var(
            self=self,
            filter_size=1,
            in_channels=out_channels * channel_multiplier,
            out_channels=out_channels,
            name='%s_H_gate_point_%s' % (name, 'z'))

        # R
        self, x_r_depth_filters, _ = ff.get_conv_var(
            self=self,
            filter_size=gate_filter_size,
            in_channels=in_channels,
            out_channels=channel_multiplier,
            name='%s_X_gate_depth_%s' % (name, 'r'))
        self, x_r_point_filters, _ = ff.get_conv_var(
            self=self,
            filter_size=1,
            in_channels=in_channels * channel_multiplier,
            out_channels=out_channels,
            name='%s_X_gate_point_%s' % (name, 'r'))
        self, h_r_depth_filters, _ = ff.get_conv_var(
            self=self,
            filter_size=gate_filter_size,
            in_channels=out_channels,  # For the hidden state
            out_channels=channel_multiplier,
            name='%s_H_gate_depth_%s' % (name, 'r'))
        self, h_r_point_filters, h_r_point_bias = ff.get_conv_var(
            self=self,
            filter_size=1,
            in_channels=out_channels * channel_multiplier,
            out_channels=out_channels,
            name='%s_H_gate_point_%s' % (name, 'r'))

        # Split off last h weight
        self, h_depth_filter, _ = ff.get_conv_var(
            self=self,
            filter_size=filter_size,
            in_channels=out_channels,  # For the hidden state
            out_channels=channel_multiplier,
            name='%s_H_gate_depth_%s' % (name, 'h'))
        self, h_point_filter, _ = ff.get_conv_var(
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
        train_alexnet, init_bias = True, False
        if 'trainable' in aux.keys():
            train_alexnet = aux['trainable']
        if 'init_bias' in aux.keys():
            init_bias = aux['init_bias']
        assert out_channels == alexnet_filter.shape[-1],\
            'Set weights = %s.' % alexnet_filter.shape[-1]
        if in_channels < alexnet_filter.shape[-2] and in_channels == 1:
            alexnet_filter = np.mean(alexnet_filter, axis=2, keepdims=True)
        elif in_channels < alexnet_filter.shape[-2]:
            raise RuntimeError('Input features = %s, Alexnet features = %s' % (
                in_channels, alexnet_filter.shape[-2]))
        x_filter = tf.get_variable(
            name=name + "_filters",
            initializer=alexnet_filter,
            trainable=train_alexnet)
        self.var_dict[(name, 0)] = x_filter
        if init_bias:
            alexnet_bias = tf.truncated_normal([out_channels], .0, .001)
        self, h_bias = ff.get_var(
            self=self,
            initial_value=alexnet_bias,
            name=name,
            idx=1,
            var_name=name + "_biases")

        # User the FF filters to mask the sequence of images
        if 'cam_mask' in aux.keys():
            cam_mask = tf.nn.conv3d(
                bottom,
                tf.expand_dims(x_filter, axis=0),
                strides=[1, 1, 1, 1, 1],
                padding='SAME')
            cam_mask = tf.reduce_mean(
                cam_mask, reduction_indices=[1, 4], keep_dims=True)
            cam_min = tf.reduce_min(
                cam_mask, reduction_indices=[2, 3], keep_dims=True)
            cam_max = tf.reduce_max(
                cam_mask, reduction_indices=[2, 3], keep_dims=True)
            cam_mask = (cam_mask - cam_min) / (cam_max - cam_min)
            bottom *= cam_mask

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
            self, iW, ib = ff.get_conv_var(
                self=self,
                filter_size=fs,
                in_channels=in_channels,  # For the hidden state
                out_channels=out_channels,
                name='%s_X_gate_%s' % (name, g))
            x_weights += [iW]
            biases += [ib]
            if idx != len(gates):
                self, iW, ib = ff.get_conv_var(
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
        self, h_filter, h_bias = ff.get_conv_var(
            self=self,
            filter_size=filter_size,
            in_channels=out_channels,  # For the hidden state
            out_channels=out_channels,
            name='%s_H_gate_%s' % (name, 'h'))
        self, x_filter, _ = ff.get_conv_var(
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
            self, iW, ib = ff.get_conv_var(
                self=self,
                filter_size=fs,
                in_channels=in_channels,  # For the hidden state
                out_channels=out_channels,
                name='%s_X_gate_%s' % (name, g))
            x_weights += [iW]
            biases += [ib]
            if idx != len(gates):
                self, iW, ib = ff.get_conv_var(
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
        self, h_filter, h_bias = ff.get_conv_var(
            self=self,
            filter_size=filter_size,
            in_channels=out_channels,  # For the hidden state
            out_channels=out_channels,
            name='%s_H_gate_%s' % (name, 'h'))
        self, x_filter, _ = ff.get_conv_var(
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
            _, iW, ib = ff.get_conv_var(
                self=self,
                filter_size=fs,
                in_channels=in_channels,  # For the hidden state
                out_channels=out_channels,
                name='%s_X_gate_%s' % (name, g))
            x_weights += [iW]
            biases += [ib]
            if idx != len(gates):
                _, iW, ib = ff.get_conv_var(
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
        self, h_filter, h_bias = ff.get_conv_var(
            self=self,
            filter_size=filter_size,
            in_channels=out_channels,  # For the hidden state
            out_channels=out_channels,
            name='%s_H_gate_%s' % (name, 'h'))
        self, x_filter, _ = ff.get_conv_var(
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
        self, h_filter, h_bias = ff.get_conv_var(
            self=self,
            filter_size=filter_size,
            in_channels=out_channels,  # For the hidden state
            out_channels=out_channels,
            # init_type='identity',  # https://arxiv.org/abs/1504.00941
            name='%s_H_gate_%s' % (name, 'h'))
        self, x_filter, _ = ff.get_conv_var(
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
