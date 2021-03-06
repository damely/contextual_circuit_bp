"""Contextual model with partial filters."""
import numpy as np
import tensorflow as tf
from utils import py_utils
from ops import initialization


# Dependency for symmetric weight ops is in models/layers/ff.py
def auxilliary_variables():
    """A dictionary containing defaults for auxilliary variables.

    These are adjusted by a passed aux dict variable."""
    return {
        'lesions': [None],  # ['Q', 'T', 'P', 'U'],
        'dtype': tf.float32,
        'return_weights': True,
        'hidden_init': 'zeros',
        'tuning_init': 'cov',  # TODO: Initialize tuning as input covariance
        'association_field': False,
        'tuning_nl': tf.nn.relu,
        'dropout': None,
        'separable': False,  # Need C++ implementation.
        'recurrent_nl': tf.nn.tanh,  # tf.nn.leakyrelu, tf.nn.relu, tf.nn.selu
        'gate_nl': tf.nn.sigmoid,
        'normal_initializer': False,
        'symmetric_weights': True,  # Lateral weight sharing
        'gru_gates': False,  # True input reset gate vs. integration gate
        'post_tuning_nl': False,  # Apply nonlinearity to crf/ecrf activity
        'gate_filter': 1,  # Gate kernel size
        'zeta': False,  # Scale I
        'gamma': False,  # Scale P
        'delta': False,  # Scale Q
        'xi': False,  # Scale X
        'vgg_extentions': 3,
        'dense_connections': False,
        'multiplicative_excitation': True,
        'learn_crf_spatial': False,
        'rectify_weights': False  # +/- rectify weights or activities
    }


class ContextualCircuit(object):
    def __getitem__(self, name):
        return getattr(self, name)

    def __contains__(self, name):
        return hasattr(self, name)

    def __init__(
            self,
            X,
            timesteps=1,
            SRF=1,
            SSN=9,
            SSF=29,
            strides=[1, 1, 1, 1],
            padding='SAME',
            aux=None,
            train=True):
        """Global initializations and settings."""
        self.X = X
        self.n, self.h, self.w, self.k = [int(x) for x in X.get_shape()]
        self.timesteps = timesteps
        self.strides = strides
        self.padding = padding
        self.train = train

        # Sort through and assign the auxilliary variables
        aux_vars = auxilliary_variables()
        if aux is not None and isinstance(aux, dict):
            for k, v in aux.iteritems():
                aux_vars[k] = v
        self.update_params(aux_vars)

        # Kernel shapes
        self.SRF, self.SSN, self.SSF = SRF, SSN, SSF
        self.SSN_ext = 2 * py_utils.ifloor(SSN / 2.0) + 1
        self.SSF_ext = 2 * py_utils.ifloor(SSF / 2.0) + 1
        if self.SSN is None:
            self.SSN = self.SRF * 3
        if self.SSF is None:
            self.SSF = self.SRF * 5
        if self.separable:
            self.q_shape = [self.SRF, self.SRF, 1, 1]
            self.u_shape = [self.SRF, self.SRF, 1, 1]
            self.p_shape = [self.SSF_ext, self.SSF_ext, 1, 1]
        else:
            self.q_shape = [self.SRF, self.SRF, self.k, self.k]
            self.u_shape = [self.SRF, self.SRF, self.k, 1]
            self.p_shape = [self.SSF_ext, self.SSF_ext, self.k, self.k]
        if self.vgg_extentions > 1:
            self.p_shape[0] /= self.vgg_extentions
            self.p_shape[1] /= self.vgg_extentions
        self.i_shape = [self.gate_filter, self.gate_filter, self.k, self.k]
        self.o_shape = [self.gate_filter, self.gate_filter, self.k, self.k]
        self.bias_shape = [1, 1, 1, self.k]
        self.tuning_params = ['Q', 'P']  # Learned connectivity
        self.tuning_shape = [1, 1, self.k, self.k]

        # Nonlinearities and initializations
        self.u_nl = tf.identity
        self.q_nl = tf.identity
        self.p_nl = tf.identity

    def update_params(self, kwargs):
        """Update the class attributes with kwargs."""
        if kwargs is not None:
            for k, v in kwargs.iteritems():
                setattr(self, k, v)

    def symmetric_weights(self, w, name):
        """Apply symmetric weight sharing."""
        conv_w_t = tf.transpose(w, (2, 3, 0, 1))
        conv_w_symm = 0.5 * (conv_w_t + tf.transpose(conv_w_t, (1, 0, 2, 3)))
        conv_w = tf.transpose(conv_w_symm, (2, 3, 0, 1), name=name)
        return conv_w

    def prepare_tensors(self):
        """ Prepare recurrent/forward weight matrices."""
        self.weight_dict = {  # Weights lower/activity upper
            'U': {
                'r': {
                    'weight': 'u_r',
                    'activity': 'U_r'
                    }
                },
            'P': {
                'r': {
                    'weight': 'p_r',
                    'activity': 'P_r',
                    'tuning': 'p_t'
                    }
                },
            'Q': {
                'r': {
                    'weight': 'q_r',
                    'activity': 'Q_r',
                    'tuning': 'q_t'
                    }
                },
            'I': {
                'r': {  # Recurrent state
                    'weight': 'i_r',
                    'bias': 'i_b',
                    'activity': 'I_r'
                },
                'f': {  # Recurrent state
                    'weight': 'i_f',
                    'activity': 'I_f'
                },
            },
            'O': {
                'r': {  # Recurrent state
                    'weight': 'o_r',
                    'bias': 'o_b',
                    'activity': 'O_r'
                },
                'f': {  # Recurrent state
                    'weight': 'o_f',
                    'activity': 'O_f'
                },
            },
            'xi': {
                'r': {  # Recurrent state
                    'weight': 'xi',
                }
            },
            'alpha': {
                'r': {  # Recurrent state
                    'weight': 'alpha',
                }
            },
            'beta': {
                'r': {  # Recurrent state
                    'weight': 'beta',
                }
            },
            'mu': {
                'r': {  # Recurrent state
                    'weight': 'mu',
                }
            },
            'nu': {
                'r': {  # Recurrent state
                    'weight': 'nu',
                }
            },
            'zeta': {
                'r': {  # Recurrent state
                    'weight': 'zeta',
                }
            },
            'gamma': {
                'r': {  # Recurrent state
                    'weight': 'gamma',
                }
            },
            'delta': {
                'r': {  # Recurrent state
                    'weight': 'delta',
                }
            }
        }

        # tuned summation: pooling in h, w dimensions
        #############################################
        q_array = np.ones(self.q_shape) / np.prod(self.q_shape)
        if 'Q' in self.lesions:
            q_array = np.zeros_like(q_array).astype(np.float32)
            print 'Lesioning CRF excitation.'
        if self.learn_crf_spatial and 'Q' not in self.lesions:
            setattr(
                self,
                self.weight_dict['Q']['r']['weight'],
                tf.get_variable(
                    name=self.weight_dict['Q']['r']['weight'],
                    dtype=self.dtype,
                    initializer=initialization.xavier_initializer(
                        shape=self.q_shape,
                        uniform=self.normal_initializer),
                    trainable=True))
        else:
            setattr(
                self,
                self.weight_dict['Q']['r']['weight'],
                tf.get_variable(
                    name=self.weight_dict['Q']['r']['weight'],
                    dtype=self.dtype,
                    initializer=q_array.astype(np.float32),
                    trainable=False))

        # weakly tuned summation: pooling in h, w dimensions
        #############################################
        p_array = np.ones(self.p_shape)
        p_array[
            self.SSN // 2 - py_utils.ifloor(
                self.SRF / 2.0):self.SSF // 2 + py_utils.iceil(
                self.SSN / 2.0),
            self.SSN // 2 - py_utils.ifloor(
                self.SRF / 2.0):self.SSF // 2 + py_utils.iceil(
                self.SSN / 2.0),
            :,  # exclude CRF!
            :] = 0.0
        p_array = p_array / p_array.sum()
        if 'P' in self.lesions:
            print 'Lesioning near eCRF.'
            p_array = np.zeros_like(p_array).astype(np.float32)

        # Association field is fully learnable
        if 'P' not in self.lesions and self.association_field:
            if self.vgg_extentions <= 1:
                setattr(
                    self,
                    self.weight_dict['P']['r']['weight'],
                    tf.get_variable(
                        name=self.weight_dict['P']['r']['weight'],
                        dtype=self.dtype,
                        initializer=initialization.xavier_initializer(
                            shape=self.p_shape,
                            uniform=self.normal_initializer),
                        trainable=True))
            else:
                for pidx in range(self.vgg_extentions):
                    if pidx == 0:
                        it_key = self.weight_dict['P']['r']['weight']
                    else:
                        self.weight_dict[
                            'P']['r']['weight_%s' % pidx] = 'p_r_%s' % pidx
                        it_key = self.weight_dict['P']['r']['weight_%s' % pidx]
                    setattr(
                        self,
                        it_key,
                        tf.get_variable(
                            name=it_key,
                            dtype=self.dtype,
                            initializer=initialization.xavier_initializer(
                                shape=self.p_shape,
                                uniform=self.normal_initializer),
                            trainable=True))
        else:
            setattr(
                self,
                self.weight_dict['P']['r']['weight'],
                tf.get_variable(
                    name=self.weight_dict['P']['r']['weight'],
                    dtype=self.dtype,
                    initializer=p_array.astype(np.float32),
                    trainable=False))

        # Connectivity tensors -- Q/P/T
        if 'Q' in self.lesions:
            print 'Lesioning CRF excitation connectivity.'
            setattr(
                self,
                self.weight_dict['Q']['r']['tuning'],
                tf.get_variable(
                    name=self.weight_dict['Q']['r']['tuning'],
                    dtype=self.dtype,
                    trainable=False,
                    initializer=np.zeros(
                        self.tuning_shape).astype(np.float32)))
        else:
            setattr(
                self,
                self.weight_dict['Q']['r']['tuning'],
                tf.get_variable(
                    name=self.weight_dict['Q']['r']['tuning'],
                    dtype=self.dtype,
                    trainable=True,
                    initializer=initialization.xavier_initializer(
                        shape=self.tuning_shape,
                        uniform=self.normal_initializer,
                        mask=None)))
        if not self.association_field:
            # Need a tuning tensor for near surround
            if 'P' in self.lesions:
                print 'Lesioning near eCRF connectivity.'
                setattr(
                    self,
                    self.weight_dict['P']['r']['tuning'],
                    tf.get_variable(
                        name=self.weight_dict['P']['r']['tuning'],
                        dtype=self.dtype,
                        trainable=False,
                        initializer=np.zeros(
                            self.tuning_shape).astype(np.float32)))
            else:
                setattr(
                    self,
                    self.weight_dict['P']['r']['tuning'],
                    tf.get_variable(
                        name=self.weight_dict['P']['r']['tuning'],
                        dtype=self.dtype,
                        trainable=True,
                        initializer=initialization.xavier_initializer(
                            shape=self.tuning_shape,
                            uniform=self.normal_initializer,
                            mask=None)))
        # Input
        setattr(
            self,
            self.weight_dict['I']['r']['weight'],
            tf.get_variable(
                name=self.weight_dict['I']['r']['weight'],
                dtype=self.dtype,
                trainable=True,
                initializer=initialization.xavier_initializer(
                    shape=self.i_shape,
                    uniform=self.normal_initializer,
                    mask=None)))
        setattr(
            self,
            self.weight_dict['I']['f']['weight'],
            tf.get_variable(
                name=self.weight_dict['I']['f']['weight'],
                dtype=self.dtype,
                trainable=True,
                initializer=initialization.xavier_initializer(
                    shape=self.i_shape,
                    uniform=self.normal_initializer,
                    mask=None)))
        setattr(
            self,
            self.weight_dict['I']['r']['bias'],
            tf.get_variable(
                name=self.weight_dict['I']['r']['bias'],
                dtype=self.dtype,
                trainable=True,
                initializer=tf.ones(self.bias_shape)))

        # Output
        setattr(
            self,
            self.weight_dict['O']['r']['weight'],
            tf.get_variable(
                name=self.weight_dict['O']['r']['weight'],
                dtype=self.dtype,
                trainable=True,
                initializer=initialization.xavier_initializer(
                    shape=self.o_shape,
                    uniform=self.normal_initializer,
                    mask=None)))
        setattr(
            self,
            self.weight_dict['O']['f']['weight'],
            tf.get_variable(
                name=self.weight_dict['O']['f']['weight'],
                dtype=self.dtype,
                trainable=True,
                initializer=initialization.xavier_initializer(
                    shape=self.o_shape,
                    uniform=self.normal_initializer,
                    mask=None)))
        # TODO: Combine Gates
        setattr(  # TODO: smart initialization of these
            self,
            self.weight_dict['O']['r']['bias'],
            tf.get_variable(
                name=self.weight_dict['O']['r']['bias'],
                dtype=self.dtype,
                trainable=True,
                initializer=tf.ones(self.bias_shape)))

        # Vector weights
        w_array = np.ones([1, 1, 1, self.k]).astype(np.float32)
        b_array = np.zeros([1, 1, 1, self.k]).astype(np.float32)

        # Divisive params
        self.alpha = tf.get_variable(name='alpha', initializer=w_array)
        self.beta = tf.get_variable(name='beta', initializer=w_array)

        # Subtractive params
        self.mu = tf.get_variable(name='mu', initializer=b_array)
        self.nu = tf.get_variable(name='nu', initializer=b_array)
        if self.zeta:
            self.zeta = tf.get_variable(name='zeta', initializer=w_array)
        else:
            self.zeta = tf.constant(1.)
        if self.gamma:
            self.gamma = tf.get_variable(name='gamma', initializer=w_array)
        else:
            self.gamma = tf.constant(1.)
        if self.delta:
            self.delta = tf.get_variable(name='delta', initializer=w_array)
        else:
            self.delta = tf.constant(1.)
        if self.xi:
            self.xi = tf.get_variable(name='xi', initializer=w_array)
        else:
            self.xi = tf.constant(1.)

    def conv_2d_op(
            self,
            data,
            weight_key,
            out_key=None,
            weights=None,
            symmetric_weights=False,
            rectify=None):
        """2D convolutions, lesion, return or assign activity as attribute."""
        if weights is None:
            weights = self[weight_key]
        if rectify is not None:
            weights = rectify(weights, 0)
        w_shape = [int(w) for w in weights.get_shape()]
        if len(w_shape) > 1 and int(w_shape[-2]) > 1:
            # Full convolutions
            if symmetric_weights:
                g = tf.get_default_graph()
                with g.gradient_override_map({'Conv2D': 'SymmetricConv'}):
                    activities = tf.nn.conv2d(
                        data,
                        weights,
                        self.strides,
                        padding=self.padding)
            else:
                activities = tf.nn.conv2d(
                    data,
                    weights,
                    self.strides,
                    padding=self.padding)
        elif len(w_shape) > 1 and int(w_shape[-2]) == 1:
            # Separable spacial
            d = int(data.get_shape()[-1])
            split_data = tf.split(data, d, axis=3)
            sep_convs = []
            for idx in range(len(split_data)):
                # TODO: Write the c++ for this.
                if self.symmetric_weights:
                    g = tf.get_default_graph()
                    with g.gradient_override_map({'Conv2D': 'SymmetricConv'}):
                        sep_convs += [tf.nn.conv2d(
                            split_data[idx],
                            weights,
                            self.strides,
                            padding=self.padding)]
                else:
                    sep_convs += [tf.nn.conv2d(
                        split_data[idx],
                        weights,
                        self.strides,
                        padding=self.padding)]
            activities = tf.concat(sep_convs, axis=-1)
        else:
            raise RuntimeError

        # Do a split convolution
        if out_key is None:
            return activities
        else:
            setattr(
                self,
                out_key,
                activities)

    def apply_tuning(self, data, wm, nl=False, rectify=None, symmetric_weights=True):
        """Wrapper for applying weight wm to data."""
        for k in self.tuning_params:
            if wm == k:
                data = self.conv_2d_op(
                    data=data,
                    weight_key=self.weight_dict[wm]['r']['tuning'],
                    rectify=rectify,
                    symmetric_weights=symmetric_weights)
                if nl:
                    return self.tuning_nl(data)
                else:
                    return data
        return data

    def zoneout(self, dropout):
        """Calculate a dropout mask for update gates."""
        return tf.cast(
            tf.greater(tf.random_uniform(
                [1, 1, 1, self.k],
                minval=0,
                maxval=1.),
                dropout),  # zone-out dropout mask
            tf.float32)

    def process_p(self, data, key, rectification):
        """Wrapper for surorund operations."""
        p_weights = self[key]
        if self.rectify_weights:
            p_weights = rectification(p_weights, 0)
        P = self.conv_2d_op(
            data=data,
            weight_key=key,
            weights=p_weights,
            symmetric_weights=self.symmetric_weights)
        return P

    def full(self, i0, O, I):
        """Contextual circuit body."""

        # -- Circuit input receives recurrent output (O)
        # CRF Excitation
        if self.association_field:
            # Use a full kernel for - surround.
            if self.vgg_extentions > 1:
                # Allow for a vgg-style calculation of surround
                previous_P = []
                for pidx in range(self.vgg_extentions):
                    if pidx == 0:
                        it_key = self.weight_dict['P']['r']['weight']
                        P = self.process_p(
                            data=O,
                            key=it_key,
                            rectification=tf.minimum)
                    else:
                        # Skip connections between surround subfilters
                        if self.dense_connections:
                            previous_P += [P]
                        it_key = self.weight_dict['P']['r']['weight_%s' % pidx]
                        P = self.process_p(
                            data=P,
                            key=it_key,
                            rectification=tf.minimum)
                        if self.dense_connections:
                            for dense_p in previous_P:
                                P = P + dense_p
            else:
                # Use a single filter for the surround
                P = self.process_p(
                    data=O,
                    key=self.weight_dict['P']['r']['weight'],
                    rectification=tf.minimum)
        else:
            # Use 1x1 convolution surround tuning
            P = self.conv_2d_op(
                data=self.apply_tuning(
                    data=O,
                    wm='P',
                    nl=self.post_tuning_nl,
                    rectify=tf.minimum,
                    symmetric_weights=self.symmetric_weights),
                weight_key=self.weight_dict['P']['r']['weight'])

        # Rectify surround activities instead of weights
        if not self.rectify_weights:
            P = tf.minimum(P, 0)

        if self.learn_crf_spatial:
            # Set the CRF to a learnable shared +/- kernel
            q_weights = self[
                self.weight_dict['Q']['r']['weight']]
            if self.rectify_weights:
                q_weights = tf.minimum(q_weights, 0)
            Q = self.conv_2d_op(
                data=I,
                weight_key=self.weight_dict['Q']['r']['weight'],
                weights=q_weights,
                symmetric_weights=self.symmetric_weights)
        else:
            # Use 1x1 convolution CRF tuning
            Q = self.conv_2d_op(
                data=self.apply_tuning(
                    data=I,
                    wm='Q',
                    nl=self.post_tuning_nl,
                    rectify=tf.minimum,
                    symmetric_weights=self.symmetric_weights),
                weight_key=self.weight_dict['Q']['r']['weight'])

        # Rectify CRF activities instead of weights
        if not self.rectify_weights:
            Q = tf.minimum(Q, 0)

        # Input gates
        I_update_input = self.conv_2d_op(
            data=self.X,
            weight_key=self.weight_dict['I']['f']['weight'])
        I_update_recurrent = self.conv_2d_op(
            data=O,
            weight_key=self.weight_dict['I']['r']['weight'])
        I_update = self.gate_nl(
            I_update_input + I_update_recurrent + self[
                self.weight_dict['I']['r']['bias']])

        # Calculate and apply dropout if requested
        if self.train and self.dropout is not None:
            I_update = self.zoneout(self.dropout) * self.gate_nl(
                I_update_input + I_update_recurrent)
        elif not self.train and self.dropout is not None:
            I_update = (1 / self.dropout) * self.gate_nl(
                I_update_input + I_update_recurrent)

        # Calculate input (-) integration
        if self.gru_gates:
            # GRU_gates applied to I before integration
            I = I_update * I
        I_summand = self.recurrent_nl(
            (self.xi * self.X)
            - ((self.alpha * I + self.mu) * Q)
            - ((self.beta * I + self.nu) * P))
        if not self.gru_gates:
            # Alternatively, forget gates on the input
            I = (I_update * I) + ((1 - I_update) * I_summand)
        else:
            I = I_summand

        # -- Circuit output receives recurrent input (I)
        if self.association_field:
            # Use a full kernel for + surround.
            if self.vgg_extentions > 1:
                # Allow for a vgg-style calculation of surround
                previous_P = []
                for pidx in range(self.vgg_extentions):
                    if pidx == 0:
                        it_key = self.weight_dict['P']['r']['weight']
                        P = self.process_p(
                            data=I,
                            key=it_key,
                            rectification=tf.maximum)
                    else:
                        # Skip connections between surround subfilters
                        if self.dense_connections:
                            previous_P += [P]
                        it_key = self.weight_dict['P']['r']['weight_%s' % pidx]
                        P = self.process_p(
                            data=P,
                            key=it_key,
                            rectification=tf.maximum)
                        if self.dense_connections:
                            for dense_p in previous_P:
                                P = P + dense_p
            else:
                # Use a single filter for the surround
                P = self.process_p(
                    data=I,
                    key=self.weight_dict['P']['r']['weight'],
                    rectification=tf.maximum)
        else:
            # Use 1x1 convolution surround tuning
            P = self.conv_2d_op(
                data=self.apply_tuning(
                    data=O,
                    wm='P',
                    nl=self.post_tuning_nl,
                    rectify=tf.minimum,
                    symmetric_weights=self.symmetric_weights),
                weight_key=self.weight_dict['P']['r']['weight'])

        # Rectify surround activities instead of weights
        if not self.rectify_weights:
            P = tf.maximum(P, 0)

        # CRF excitation
        if self.learn_crf_spatial:
            # Set the CRF to a learnable shared +/- kernel
            q_weights = self[
                self.weight_dict['Q']['r']['weight']]
            if self.rectify_weights:
                q_weights = tf.maximum(q_weights, 0)
            Q = self.conv_2d_op(
                data=I,
                weight_key=self.weight_dict['Q']['r']['weight'],
                weights=q_weights,
                symmetric_weights=self.symmetric_weights)
        else:
            # Use 1x1 convolution CRF tuning
            Q = self.conv_2d_op(
                data=self.apply_tuning(
                    data=I,
                    wm='Q',
                    nl=self.post_tuning_nl,
                    rectify=tf.maximum,
                    symmetric_weights=self.symmetric_weights),
                weight_key=self.weight_dict['Q']['r']['weight'])

        # Rectify surround activities instead of weights
        if not self.rectify_weights:
            Q = tf.maximum(Q, 0)

        # Output gates
        O_update_input = self.conv_2d_op(
            data=self.X,
            weight_key=self.weight_dict['O']['f']['weight'])
        O_update_recurrent = self.conv_2d_op(
            data=I,
            weight_key=self.weight_dict['O']['r']['weight'])
        O_update = self.gate_nl(
            O_update_input + O_update_recurrent + self[
                self.weight_dict['O']['r']['bias']])

        # Calculate and apply dropout if requested
        if self.train and self.dropout is not None:
            O_update = self.zoneout(self.dropout) * self.gate_nl(
                O_update_input + O_update_recurrent)
        elif not self.train and self.dropout is not None:
            O_update = (1 / self.dropout) * self.gate_nl(
                O_update_input + O_update_recurrent)

        # Calculate output (+) integration
        if self.multiplicative_excitation:
            # Multiplicative gating I * (P + Q)
            O_summand = self.recurrent_nl(
                self.zeta * I * ((self.gamma * P) + (self.delta * Q)))
        else:
            # Additive gating I + P + Q
            O_summand = self.recurrent_nl(
                self.zeta * I + self.gamma * P + self.delta * Q)
        O = (O_update * O) + ((1 - O_update) * O_summand)
        i0 += 1  # Iterate loop
        return i0, O, I

    def condition(self, i0, O, I):
        """While loop halting condition."""
        return i0 < self.timesteps

    def gather_tensors(self, wak='weight'):
        weights = {}
        for k, v in self.weight_dict.iteritems():
            for wk, wv in v.iteritems():
                if wak in wv.keys() and hasattr(self, wv[wak]):
                    weights['%s_%s' % (k, wk)] = self[wv[wak]]

        return weights

    def build(self, reduce_memory=False):
        """Run the backprop version of the CCircuit."""
        self.prepare_tensors()
        i0 = tf.constant(0)
        if self.hidden_init == 'identity':
            I = tf.identity(self.X)
            O = tf.identity(self.X)
        elif self.hidden_init == 'random':
            I = initialization.xavier_initializer(
                shape=[self.n, self.h, self.w, self.k],
                uniform=self.normal_initializer,
                mask=None)
            O = initialization.xavier_initializer(
                shape=[self.n, self.h, self.w, self.k],
                uniform=self.normal_initializer,
                mask=None)
        elif self.hidden_init == 'zeros':
            I = tf.zeros_like(self.X)
            O = tf.zeros_like(self.X)
        else:
            raise RuntimeError

        if reduce_memory:
            print 'Warning: Using FF version of the model.'
            for t in range(self.timesteps):
                i0, O, I = self.full(i0, O, I)
        else:
            # While loop
            elems = [
                i0,
                O,
                I
            ]
            returned = tf.while_loop(
                self.condition,
                self.full,
                loop_vars=elems,
                back_prop=True,
                swap_memory=False)

            # Prepare output
            i0, O, I = returned  # i0, O, I

        if self.return_weights:
            weights = self.gather_tensors(wak='weight')
            tuning = self.gather_tensors(wak='tuning')
            new_tuning = {}
            for k, v in tuning.iteritems():
                key_name = v.name.split('/')[-1].split(':')[0]
                new_tuning[key_name] = v
            weights = dict(weights, **new_tuning)
            activities = self.gather_tensors(wak='activity')
            # Attach weights if using association field
            if self.association_field:
                weights['p_t'] = self.p_r  # Make available for regularization
            return O, weights, activities
        else:
            return O
