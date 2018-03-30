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
        'hidden_init': 'identity',
        'association_field': True,
        'tuning_nl': tf.nn.relu,
        'store_states': False,
        'train': True,
        'dropout': None,
        # 'separable': False,  # Need C++ implementation.
        'recurrent_nl': tf.nn.tanh,  # tf.nn.leaky_relu, tf.nn.relu, tf.nn.selu
        'gate_nl': tf.nn.sigmoid,
        'ecrf_nl': tf.nn.relu,
        'normal_initializer': True,
        'symmetric_weights': False,  # Lateral weight sharing
        'symmetric_gate_weights': False,
        'gru_gates': False,  # True input reset gate vs. integration gate
        'post_tuning_nl': tf.nn.relu,  # Nonlinearity on crf activity
        'gate_filter': 1,  # Gate kernel size
        'zeta': True,  # Scale I (excitatory state)
        'gamma': True,  # Scale P
        'delta': True,  # Scale Q
        'xi': True,  # Scale X
        'alpha': True,  # divisive CRF
        'beta': True,  # divisive eCRF
        'mu': True,  # subtractive CRF
        'nu': True,  # subtractive eCRF
        'batch_norm': False,
        'integration_type': 'alternate',  # Psych review (mely) or alternate
        'dense_connections': False,  # Dense connections on VGG-style convs
        'atrous_convolutions': False,  # Non-zero integer controls rate
        'multiplicative_excitation': False,
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

        # if isinstance(SSN, list):
        #     self.SSN_ext = [2 * py_utils.ifloor(x / 2.0) + 1 for x in SSN]
        # else:
        #     self.SSN_ext = 2 * py_utils.ifloor(SSN / 2.0) + 1
        if isinstance(SSF, list):
            self.SSF_ext = [2 * py_utils.ifloor(x / 2.0) + 1 for x in SSF]
        else:
            self.SSF_ext = 2 * py_utils.ifloor(SSF / 2.0) + 1
        if self.SSN is None:
            self.SSN = self.SRF * 3
        if self.SSF is None:
            self.SSF = self.SRF * 5

        # if self.separable:
        #     self.q_shape = [self.SRF, self.SRF, 1, 1]
        #     self.u_shape = [self.SRF, self.SRF, 1, 1]
        #     self.p_shape = [self.SSF_ext, self.SSF_ext, 1, 1]
        self.q_shape = [self.SRF, self.SRF, self.k, self.k]
        self.u_shape = [self.SRF, self.SRF, self.k, 1]
        if isinstance(SSF, list):
            self.p_shape = [
                [ssf_ext, ssf_ext, self.k, self.k] for ssf_ext in self.SSF_ext]
        else:
            self.p_shape = [self.SSF_ext, self.SSF_ext, self.k, self.k]
        self.i_shape = [self.gate_filter, self.gate_filter, self.k, self.k]
        self.o_shape = [self.gate_filter, self.gate_filter, self.k, self.k]
        self.bias_shape = [1, 1, 1, self.k]
        self.tuning_params = ['Q', 'P']  # Learned connectivity
        self.tuning_shape = [1, 1, self.k, self.k]

        # Nonlinearities and initializations
        self.u_nl = tf.identity
        self.q_nl = tf.identity
        self.p_nl = tf.identity

        # Set integration operations
        self.ii, self.oi = self.interpret_integration(self.integration_type)

    def interpret_integration(self, integration_type):
        """Return function for integration."""
        if integration_type == 'mely':
            return self.mely_input_integration, self.mely_output_integration
        elif integration_type == 'alternate':
            return self.input_integration, self.output_integration
        else:
            raise NotImplementedError(
                'Requested integration %s' % integration_type)

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
            },
            'O': {
                'r': {  # Recurrent state
                    'weight': 'o_r',
                    'bias': 'o_b',
                    'activity': 'O_r'
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
            'phi': {
                'r': {  # Recurrent state
                    'weight': 'phi',
                }
            },
            'kappa': {
                'r': {  # Recurrent state
                    'weight': 'kappa',
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
        setattr(
            self,
            self.weight_dict['Q']['r']['weight'],
            tf.get_variable(
                name=self.weight_dict['Q']['r']['weight'],
                dtype=self.dtype,
                initializer=q_array.astype(np.float32),
                trainable=False))

        # untuned suppression: reduction across feature axis
        ####################################################
        u_array = np.ones(self.u_shape) / np.prod(self.u_shape)
        if 'U' in self.lesions:
            u_array = np.zeros_like(u_array).astype(np.float32)
            print 'Lesioning CRF inhibition.'
        setattr(
            self,
            self.weight_dict['U']['r']['weight'],
            tf.get_variable(
                name=self.weight_dict['U']['r']['weight'],
                dtype=self.dtype,
                initializer=u_array.astype(np.float32),
                trainable=False))

        # weakly tuned summation: pooling in h, w dimensions
        #############################################
        if isinstance(self.p_shape[0], list) and 'P' not in self.lesions:
            # VGG-style filters
            for pidx, pext in enumerate(self.p_shape):
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
                            shape=pext,
                            uniform=self.normal_initializer),
                        trainable=True))
        else:
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
            if self.association_field and 'P' not in self.lesions:
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
        # Gate weights
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
        setattr(  # TODO: smart initialization of these
            self,
            self.weight_dict['O']['r']['bias'],
            tf.get_variable(
                name=self.weight_dict['O']['r']['bias'],
                dtype=self.dtype,
                trainable=True,
                initializer=tf.ones(self.bias_shape)))

        # Degree of freedom weights (vectors)
        w_shape = [1, 1, 1, self.k]
        b_shape = [1, 1, 1, self.k]
        # w_array = np.ones(w_shape).astype(np.float32)
        # b_array = np.zeros(b_shape).astype(np.float32)

        # Divisive params
        if self.alpha:
            self.alpha = tf.get_variable(
                name='alpha',
                initializer=initialization.xavier_initializer(
                    shape=w_shape,
                    uniform=self.normal_initializer,
                    mask=None))
        else:
            self.alpha = tf.constant(1.)
        if self.beta:
            self.beta = tf.get_variable(
                name='beta',
                initializer=initialization.xavier_initializer(
                    shape=w_shape,
                    uniform=self.normal_initializer,
                    mask=None))
        else:
            self.beta = tf.constant(1.)

        # Subtractive params
        if self.mu:
            self.mu = tf.get_variable(
                name='mu',
                initializer=initialization.xavier_initializer(
                    shape=b_shape,
                    uniform=self.normal_initializer,
                    mask=None))
        else:
            self.mu = tf.constant(1.)
        if self.nu:
            self.nu = tf.get_variable(
                name='nu',
                initializer=initialization.xavier_initializer(
                    shape=b_shape,
                    uniform=self.normal_initializer,
                    mask=None))
        else:
            self.nu = tf.constant(1.)

        if self.zeta:
            self.zeta = tf.get_variable(
                name='zeta',
                initializer=initialization.xavier_initializer(
                    shape=w_shape,
                    uniform=self.normal_initializer,
                    mask=None))
        else:
            self.zeta = tf.constant(1.)
        if self.gamma:
            self.gamma = tf.get_variable(
                name='gamma',
                initializer=initialization.xavier_initializer(
                    shape=w_shape,
                    uniform=self.normal_initializer,
                    mask=None))
        else:
            self.gamma = tf.constant(1.)
        if self.delta:
            self.delta = tf.get_variable(
                name='delta',
                initializer=initialization.xavier_initializer(
                    shape=w_shape,
                    uniform=self.normal_initializer,
                    mask=None))
        else:
            self.delta = tf.constant(1.)
        if self.xi:
            self.xi = tf.get_variable(
                name='xi',
                initializer=initialization.xavier_initializer(
                    shape=w_shape,
                    uniform=self.normal_initializer,
                    mask=None))
        else:
            self.xi = tf.constant(1.)
        if self.multiplicative_excitation:
            self.kappa = tf.get_variable(
                name='kappa',
                initializer=initialization.xavier_initializer(
                    shape=w_shape,
                    uniform=self.normal_initializer,
                    mask=None))
            self.omega = tf.get_variable(
                name='omega',
                initializer=initialization.xavier_initializer(
                    shape=w_shape,
                    uniform=self.normal_initializer,
                    mask=None))
        else:
            self.kappa = tf.constant(1.)
            self.omega = tf.constant(1.)

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
                    if self.atrous_convolutions:
                        activities = tf.nn.atrous_conv2d(
                            data,
                            weights,
                            rate=self.atrous_convolutions,
                            padding=self.padding)
                    else:
                        activities = tf.nn.conv2d(
                            data,
                            weights,
                            self.strides,
                            padding=self.padding)
            else:
                if self.atrous_convolutions:
                        activities = tf.nn.atrous_conv2d(
                            data,
                            weights,
                            rate=self.atrous_convolutions,
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

    def apply_tuning(self, data, wm, nl=False, rectify=None):
        """Wrapper for applying weight wm to data."""
        for k in self.tuning_params:
            if wm == k:
                if self.symmetric_weights:
                    g = tf.get_default_graph()
                    with g.gradient_override_map({'Conv2D': 'SymmetricConv'}):
                        data = self.conv_2d_op(
                            data=data,
                            weight_key=self.weight_dict[wm]['r']['tuning'],
                            rectify=rectify)
                else:
                    data = self.conv_2d_op(
                        data=data,
                        weight_key=self.weight_dict[wm]['r']['tuning'],
                        rectify=rectify)
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

    def p_convolution(self, data, key, rectification):
        """Apply the eCRF association field convolution."""
        p_weights = self[key]
        if self.rectify_weights:
            p_weights = rectification(p_weights, 0)
        return self.conv_2d_op(
            data=data,
            weight_key=key,
            weights=p_weights,
            symmetric_weights=self.symmetric_weights)

    def process_p(self, data, key, rectification, full=True):
        """Wrapper for eCRF operations.

        data : recurrent input or output
        key : key for the eCRF weight tensor
        rectification : if toggled, inhibition w/ tf.minimum(x, 0)
            or excitation w/ tf.maximum
        full : association-field style convolutions vs.
            1x1 tuning convolutions
        """
        if full:
            if isinstance(self.p_shape[0], list):
                P = self.hierarchical_convolutions(
                    data=data,
                    key=key,
                    rectification=rectification)
            else:
                # Do not use VGG-style convolutions
                P = self.p_convolution(
                    data=data,
                    key=key,
                    rectification=rectification)
        else:
            P = self.conv_2d_op(
                data=self.apply_tuning(
                    data=data,
                    wm='P',
                    nl=self.post_tuning_nl,
                    rectify=rectification),
                symmetric_weights=self.symmetric_weights,
                weight_key=self.weight_dict['P']['r']['weight'])
        return P

    def hierarchical_convolutions(self, data, key, rectification):
        """Approximate a full kernel with a series of smaller ones."""
        previous_P = []
        for pidx in range(len(self.SSF)):
            if pidx == 0:
                it_key = self.weight_dict['P']['r']['weight']
                P = self.p_convolution(
                    data=data,
                    key=it_key,
                    rectification=rectification)
            else:
                # Skip connections between surround subfilters
                if self.batch_norm:
                    P = tf.layers.batch_normalization(
                        P,
                        scale=True,
                        center=True,
                        training=self.train)
                if self.dense_connections:
                    previous_P += [P]
                it_key = self.weight_dict['P']['r']['weight_%s' % pidx]
                P = self.p_convolution(
                    data=P,
                    key=it_key,
                    rectification=rectification)
            if self.ecrf_nl is not None:
                P = self.ecrf_nl(P)
            if self.dense_connections:
                for dense_p in previous_P:
                    P = P + dense_p
        return P

    def circuit_input(self, O):
        """Circuit input operates on recurrent output (O)."""

        # Input gates
        I_update_recurrent = self.conv_2d_op(
            data=O,
            weight_key=self.weight_dict['I']['r']['weight'],
            symmetric_weights=self.symmetric_gate_weights)

        # Calculate and apply dropout if requested
        if self.train and self.dropout is not None:
            I_update = self.zoneout(self.dropout) * self.gate_nl(
                I_update_recurrent)
        elif not self.train and self.dropout is not None:
            I_update = (1 / self.dropout) * self.gate_nl(
                I_update_recurrent)
        else:
            I_update = self.gate_nl(
                I_update_recurrent + self[
                    self.weight_dict['I']['r']['bias']])

        if self.gru_gates:
            # GRU_gates applied to I before integration
            O *= I_update

        # eCRF Inhibition
        P = self.process_p(
            data=O,
            key=self.weight_dict['P']['r']['weight'],
            rectification=tf.minimum,
            full=self.association_field)

        # Rectify surround activities instead of weights
        if not self.rectify_weights:
            P = tf.minimum(P, 0)

        # CRF inhibitiion
        U = self.conv_2d_op(
            data=self.apply_tuning(
                data=O,
                wm='U',
                nl=self.post_tuning_nl),
            weight_key=self.weight_dict['U']['r']['weight'],
            symmetric_weights=self.symmetric_gate_weights)
        return P, U, I_update

    def circuit_output(self, I):
        """Circuit output operates on recurrent input (I)."""
        # Output gates
        O_update_recurrent = self.conv_2d_op(
            data=I,
            weight_key=self.weight_dict['O']['r']['weight'],
            symmetric_weights=self.symmetric_gate_weights)

        # Calculate and apply dropout if requested
        if self.train and self.dropout is not None:
            O_update = self.zoneout(self.dropout) * self.gate_nl(
                O_update_recurrent)
        elif not self.train and self.dropout is not None:
            O_update = (1 / self.dropout) * self.gate_nl(
                O_update_recurrent)
        else:
            O_update = self.gate_nl(
                O_update_recurrent + self[
                    self.weight_dict['O']['r']['bias']])

        # eCRF Excitation
        P = self.process_p(
            data=I,
            key=self.weight_dict['P']['r']['weight'],
            rectification=tf.maximum,
            full=self.association_field)

        # Rectify surround activities instead of weights
        if not self.rectify_weights:
            P = tf.maximum(P, 0)

        # CRF excitation
        Q = self.conv_2d_op(
            data=self.apply_tuning(
                data=I,
                wm='Q',
                nl=self.post_tuning_nl),
            weight_key=self.weight_dict['Q']['r']['weight'],
            symmetric_weights=self.symmetric_gate_weights)
        return P, Q, O_update

    def mely_input_integration(self, P, U, I, O, I_update):
        """Integration on the input."""
        I_summand = self.recurrent_nl(
            (self.xi * self.X) -
            ((self.alpha * I + self.mu) * U) -
            ((self.beta * I + self.nu) * P))
        if not self.gru_gates:
            # Alternatively, forget gates on the input
            return (I_update * I) + ((1 - I_update) * I_summand)
        else:
            return I_summand

    def mely_output_integration(self, P, Q, I, O, O_update):
        """Integration on the output."""
        if self.multiplicative_excitation:
            # Multiplicative gating I * (P + Q)
            O_summand = self.recurrent_nl(
                self.zeta * I * ((self.gamma * P) + (self.delta * Q)))
        else:
            # Additive gating I + P + Q
            O_summand = self.recurrent_nl(
                self.zeta * I + self.gamma * P + self.delta * Q)
        return (O_update * O) + ((1 - O_update) * O_summand)

    def input_integration(self, P, U, I, O, I_update):
        """Integration on the input."""
        I_summand = self.recurrent_nl(
            (self.xi * self.X) -
            ((self.alpha * O + self.mu) * U) -
            ((self.beta * O + self.nu) * P))
        if not self.gru_gates:
            # Alternatively, forget gates on the input
            return (I_update * I) + ((1 - I_update) * I_summand)
        else:
            return I_summand

    def output_integration(self, P, Q, I, O, O_update):
        """Integration on the output."""
        if self.multiplicative_excitation:
            # Multiplicative gating I * (P + Q)
            # O_summand = self.recurrent_nl(
            #     self.zeta * I * ((self.gamma * P) + (self.delta * Q)))
            # USE BOTH ADDITIVE AND MULTIPLICATIVE
            activity = self.gamma * P + self.delta * Q
            O_additive = self.kappa * (self.zeta * I + activity)
            O_multiplicative = self.omega * (self.zeta * I * activity)
            O_summand = self.recurrent_nl(O_additive + O_multiplicative)
            # USE BOTH ADDITIVE AND MULTIPLICATIVE
            # activity = self.gamma * P + self.delta * Q
            # O_additive = self.kappa * (self.zeta * I + activity)
            # O_multiplicative = (1 - self.kappa) * (self.zeta * I * activity)
            # O_summand = self.recurrent_nl(O_additive + O_multiplicative)
        else:
            # Additive gating I + P + Q
            O_summand = self.recurrent_nl(
                self.zeta * I + self.gamma * P + self.delta * Q)
        O = (O_update * O) + ((1 - O_update) * O_summand)
        return O

    def full(self, i0, O, I, store_O=None, store_I=None):
        """Contextual circuit body."""

        # -- Circuit input receives recurrent output (O)
        P, U, I_update = self.circuit_input(O)

        # Calculate input (-) integration
        I = self.ii(
            P=P,
            U=U,
            I=I,
            O=O,
            I_update=I_update)

        # -- Circuit output receives recurrent input (I)
        P, Q, O_update = self.circuit_output(I)

        # Calculate output (+) integration
        O = self.oi(
            P=P,
            Q=Q,
            I=I,
            O=O,
            O_update=O_update)

        if self.store_states:
            store_I.write(i0, I)
            store_O.write(i0, O)

        # Interate loop
        i0 += 1
        return i0, O, I, store_I, store_O

    def condition(self, i0, O, I, store_I, store_O):
        """While loop halting condition."""
        return i0 < self.timesteps

    def gather_tensors(self, wak='weight'):
        weights = {}
        for k, v in self.weight_dict.iteritems():
            for wk, wv in v.iteritems():
                if wak in wv.keys() and hasattr(self, wv[wak]):
                    weights['%s_%s' % (k, wk)] = self[wv[wak]]

        return weights

    def build(self):
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

        if self.store_states:
            store_I = tf.TensorArray(tf.float32, size=self.timesteps)
            store_O = tf.TensorArray(tf.float32, size=self.timesteps)
            elems = [
                i0,
                O,
                I,
                store_I,
                store_O
            ]
            returned = tf.while_loop(
                self.condition,
                self.full,
                loop_vars=elems,
                back_prop=True,
                swap_memory=True)

            # Prepare output
            i0, O, I, store_I, store_O = returned
            setattr(self, 'store_I', store_I.stack('store_I'))
            setattr(self, 'store_O', store_O.stack('store_O'))
        else:
            # While loop
            elems = [
                i0,
                O,
                I,
                tf.constant(0),
                tf.constant(0)
            ]
            returned = tf.while_loop(
                self.condition,
                self.full,
                loop_vars=elems,
                back_prop=True,
                swap_memory=True)

            # Prepare output
            i0, O, I, _, _ = returned

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
            if self.store_states:
                weights['store_I'] = store_I
                weights['store_O'] = store_O    
            return O, weights, activities
        else:
            if self.store_states:
                return O  # , store_I, store_O
            else:
                return O

