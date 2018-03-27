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
        'hidden_init': 'random',
        'association_field': True,
        'tuning_nl': tf.nn.relu,
        'store_states': False,
        'train': True,
        'dropout': None,
        # 'separable': False,  # Need C++ implementation.
        'recurrent_nl': tf.nn.tanh,  # tf.nn.leakyrelu, tf.nn.relu, tf.nn.selu
        'gate_nl': tf.nn.sigmoid,
        'ecrf_nl': tf.nn.relu,
        'normal_initializer': True,
        'symmetric_weights': True,  # Lateral weight sharing
        'symmetric_gate_weights': False,
        'gru_gates': False,  # True input reset gate vs. integration gate
        'post_tuning_nl': tf.nn.relu,  # Nonlinearity on crf activity
        'gate_filter': 1,  # Gate kernel size
        'zeta': True,  # Scale I (excitatory state)
        'gamma': True,  # Scale P
        'delta': True,  # Scale Q
        'xi': True,  # Scale X
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
        if 1:
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

    def circuit_input(self, O):
        """Circuit input operates on recurrent output (O)."""
        g = tf.get_default_graph()
        with g.gradient_override_map({'Conv2D': 'SymmetricConv'}):
            I_update_input = tf.nn.conv2d(
                self.X,
                self[self.weight_dict['I']['f']['weight']],
                self.strides,
                padding=self.padding)
            I_update_input = tf.nn.conv2d(
                O,
                self[self.weight_dict['I']['r']['weight']],
                self.strides,
                padding=self.padding)
            I_update = self.gate_nl(
                I_update_input + I_update_recurrent + self[
                    self.weight_dict['I']['r']['bias']])
            if self.gru_gates:
                # GRU_gates applied to I before integration
                O *= I_update
            p_weights = self[self.weight_dict['P']['r']['weight']]
            if self.rectify_weights:
                p_weights = tf.minimum(p_weights, 0)
            P = tf.nn.conv2d(
                O,
                p_weights,
                self.strides,
                padding=self.padding)

            # Rectify surround activities instead of weights
            if not self.rectify_weights:
                P = tf.minimum(P, 0)
            U = tf.nn.conv2d(
                O,
                self[self.weight_dict['U']['r']['tuning']],
                self.strides,
                padding=self.padding)
        return P, U, I_update

    def circuit_output(self, I):
        """Circuit output operates on recurrent input (I)."""
        # Output gates
        g = tf.get_default_graph()
        with g.gradient_override_map({'Conv2D': 'SymmetricConv'}):
            O_update_input = tf.nn.conv2d(
                self.X,
                self[self.weight_dict['O']['f']['weight']],
                self.strides,
                padding=self.padding)
            O_update_input = tf.nn.conv2d(
                I,
                self[self.weight_dict['O']['r']['weight']],
                self.strides,
                padding=self.padding)
            O_update = self.gate_nl(
                O_update_input + O_update_recurrent + self[
                    self.weight_dict['O']['r']['bias']])

            p_weights = self[self.weight_dict['P']['r']['weight']]
            if self.rectify_weights:
                p_weights = tf.maximum(p_weights, 0)
            P = tf.nn.conv2d(
                I,
                p_weights,
                self.strides,
                padding=self.padding)

            # Rectify surround activities instead of weights
            if not self.rectify_weights:
                P = tf.maximum(P, 0)
            Q = tf.nn.conv2d(
                I,
                self[self.weight_dict['Q']['r']['tuning']],
                self.strides,
                padding=self.padding)
        return P, Q, O_update

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
        # Additive gating I + P + Q
        O_summand = self.recurrent_nl(
            self.zeta * I + self.gamma * P + self.delta * Q)
        O = (O_update * O) + ((1 - O_update) * O_summand)
        return O

    def full(self, i0, O, I):
        """Contextual circuit body."""

        # -- Circuit input receives recurrent output (O)
        P, U, I_update = self.circuit_input(O)

        # Calculate input (-) integration
        I = self.input_integration(
            P=P,
            U=U,
            I=I,
            O=O,
            I_update=I_update)

        # -- Circuit output receives recurrent input (I)
        P, Q, O_update = self.circuit_output(I)

        # Calculate output (+) integration
        O = self.output_integration(
            P=P,
            Q=Q,
            I=I,
            O=O,
            O_update=O_update)

        # Interate loop
        i0 += 1
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
            swap_memory=True)

        # Prepare output
        i0, O, I = returned

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

