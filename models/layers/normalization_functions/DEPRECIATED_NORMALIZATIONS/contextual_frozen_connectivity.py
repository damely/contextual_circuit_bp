import numpy as np
import tensorflow as tf
from utils import py_utils
from ops import initialization


class ContextualCircuit(object):
    def __getitem__(self, name):
        return getattr(self, name)

    def __contains__(self, name):
        return hasattr(self, name)

    def __init__(
            self,
            X,
            model_version='full',
            timesteps=1,
            lesions=None,
            SRF=1,
            SSN=9,
            SSF=29,
            strides=[1, 1, 1, 1],
            padding='SAME',
            dtype=tf.float32,
            return_weights=True):

        self.X = X
        self.n, self.h, self.w, self.k = [int(x) for x in X.get_shape()]
        self.model_version = model_version
        self.timesteps = timesteps
        self.lesions = lesions
        self.strides = strides
        self.padding = padding
        self.dtype = dtype
        self.SRF, self.SSN, self.SSF = SRF, SSN, SSF

        self.SSN_ext = 2 * py_utils.ifloor(SSN / 2.0) + 1
        self.SSF_ext = 2 * py_utils.ifloor(SSF / 2.0) + 1
        self.q_shape = [self.SRF, self.SRF, self.k, self.k]
        self.u_shape = [self.SRF, self.SRF, self.k, 1]
        self.p_shape = [self.SSN_ext, self.SSN_ext, self.k, self.k]
        self.t_shape = [self.SSF_ext, self.SSF_ext, self.k, self.k]
        self.u_nl = tf.identity
        self.t_nl = tf.identity
        self.q_nl = tf.identity
        self.p_nl = tf.identity
        self.i_nl = tf.nn.relu  # input non linearity
        self.o_nl = tf.nn.relu  # output non linearity

        self.return_weights = return_weights
        self.normal_initializer = False
        if self.SSN is None:
            self.SSN = self.SRF * 3
        if self.SSF is None:
            self.SSF = self.SRF * 5

    def prepare_tensors(self):
        """ Prepare recurrent/forward weight matrices."""
        self.weight_dict = {  # Weights lower/activity upper
            'U': {
                'r': {
                    'weight': 'u_r',
                    'activity': 'U_r'
                    },
                'f': {
                    'weight': 'u_f',
                    'bias': 'ub_f',
                    'activity': 'U_f'
                    }
                },
            'T': {
                'r': {
                    'weight': 't_r',
                    'activity': 'T_r'
                    },
                'f': {
                    'weight': 't_f',
                    'bias': 'tb_f',
                    'activity': 'T_f'
                    }
                },
            'P': {
                'r': {
                    'weight': 'p_r',
                    'activity': 'P_r'
                    },
                'f': {
                    'weight': 'p_f',
                    'bias': 'pb_f',
                    'activity': 'P_f'
                    }
                },
            'Q': {
                'r': {
                    'weight': 'q_r',
                    'activity': 'Q_r'
                    },
                'f': {
                    'weight': 'q_f',
                    'bias': 'qb_f',
                    'activity': 'Q_f'
                    }
                },
            'I': {
                'r': {  # Recurrent state
                    'weight': 'i_r',
                    'activity': 'I_r'
                }
            },
            'O': {
                'r': {  # Recurrent state
                    'weight': 'o_r',
                    'activity': 'O_r'
                }
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
            },
            'eps': {
                'r': {  # Recurrent state
                    'weight': 'eps',
                }
            },
            'eta': {
                'r': {  # Recurrent state
                    'weight': 'eta',
                }
            },
            'sig': {
                'r': {  # Recurrent state
                    'weight': 'sig',
                }
            },
            'tau': {
                'r': {  # Recurrent state
                    'weight': 'tau',
                }
            },

        }

        # tuned summation: pooling in h, w dimensions
        #############################################
        setattr(
            self,
            self.weight_dict['Q']['r']['weight'],
            tf.get_variable(
                name=self.weight_dict['Q']['r']['weight'],
                dtype=self.dtype,
                initializer=initialization.xavier_initializer(
                    shape=self.q_shape,
                    uniform=self.normal_initializer,
                    mask=None)))

        # untuned suppression: reduction across feature axis
        ####################################################
        u_array = 1.0/self.k * np.ones((1, 1, self.k, 1))
        setattr(
            self,
            self.weight_dict['U']['r']['weight'],
            tf.get_variable(
                name=self.weight_dict['U']['r']['weight'],
                dtype=self.dtype,
                initializer=u_array.astype(np.float32),
                trainable=False))

        # tuned summation: pooling in h, w dimensions
        #############################################
        p_array = np.zeros(self.p_shape)
        for pdx in range(self.k):
            p_array[:self.SSN, :self.SSN, pdx, pdx] = 1.0
        p_array[
            self.SSN // 2 - py_utils.ifloor(
                self.SRF / 2.0):self.SSN // 2 + py_utils.iceil(
                self.SRF / 2.0),
            self.SSN // 2 - py_utils.ifloor(
                self.SRF / 2.0):self.SSN // 2 + py_utils.iceil(
                self.SRF / 2.0),
            :,  # exclude CRF!
            :] = 0.0

        setattr(
            self,
            self.weight_dict['P']['r']['weight'],
            tf.get_variable(
                name=self.weight_dict['P']['r']['weight'],
                dtype=self.dtype,
                initializer=p_array.astype(np.float32),
                trainable=False))

        # tuned suppression: pooling in h, w dimensions
        ###############################################
        t_array = np.zeros(self.t_shape)
        for tdx in range(self.k):
            t_array[:self.SSF, :self.SSF, tdx, tdx] = 1.0
        t_array[
            self.SSF // 2 - py_utils.ifloor(
                self.SSN / 2.0):self.SSF // 2 + py_utils.iceil(
                self.SSN / 2.0),
            self.SSF // 2 - py_utils.ifloor(
                self.SSN / 2.0):self.SSF // 2 + py_utils.iceil(
                self.SSN / 2.0),
            :,  # exclude near surround!
            :] = 0.0
        setattr(
            self,
            self.weight_dict['T']['r']['weight'],
            tf.get_variable(
                name=self.weight_dict['T']['r']['weight'],
                dtype=self.dtype,
                initializer=t_array.astype(np.float32),
                trainable=False))

        # Scalar weights
        self.xi = tf.get_variable(name='xi', initializer=1.)
        self.alpha = tf.get_variable(name='alpha', initializer=1.)
        self.beta = tf.get_variable(name='beta', initializer=1.)
        self.mu = tf.get_variable(name='mu', initializer=1.)
        self.nu = tf.get_variable(name='nu', initializer=1.)
        self.zeta = tf.get_variable(name='zeta', initializer=1.)
        self.gamma = tf.get_variable(name='gamma', initializer=1.)
        self.delta = tf.get_variable(name='delta', initializer=1.)
        self.eps = tf.get_variable(name='eps', initializer=1.)
        self.eta = tf.get_variable(name='eta', initializer=1.)
        self.sig = tf.get_variable(name='sig', initializer=1.)
        self.tau = tf.get_variable(name='tau', initializer=1.)

    def conv_2d_op(self, data, weight_key, out_key=None):
        """2D convolutions, lesion, return or assign activity as attribute."""
        if weight_key in self.lesions:
            weights = tf.constant(0.)
        else:
            weights = self[weight_key]
        activities = tf.nn.conv2d(
                data,
                weights,
                self.strides,
                padding=self.padding)
        if out_key is None:
            return activities
        else:
            setattr(
                self,
                out_key,
                activities)

    def full(self, i0, O, I):
        """Published CM with learnable weights."""
        U = self.conv_2d_op(
            data=O,
            weight_key=self.weight_dict['U']['r']['weight']
        )
        T = self.conv_2d_op(
            data=O,
            weight_key=self.weight_dict['T']['r']['weight']
        )
        P = self.conv_2d_op(
            data=I,
            weight_key=self.weight_dict['P']['r']['weight']
        )
        Q = self.conv_2d_op(
            data=I,
            weight_key=self.weight_dict['Q']['r']['weight']
        )

        eps_eta = tf.pow(self.eps, 2) * self.eta
        sig_tau = tf.pow(self.sig, 2) * self.tau

        I_summand = tf.nn.relu(
            (self.xi * self.X)
            - ((self.alpha * I + self.mu) * U)
            - ((self.beta * I + self.nu) * T))

        I = eps_eta * I + self.eta * I_summand

        O_summand = tf.nn.relu(
            self.zeta * I
            + self.gamma * P
            + self.delta * Q)
        O = sig_tau * O + self.tau * O_summand
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
        O = tf.identity(self.X)
        I = tf.identity(self.X)

        if reduce_memory:
            print 'Warning: Using FF version of the model.'
            for t in range(self.timesteps):
                i0, O, I = self[self.model_version](i0, O, I)
        else:
            # While loop
            elems = [
                i0,
                O,
                I
            ]

            returned = tf.while_loop(
                self.condition,
                self[self.model_version],
                loop_vars=elems,
                back_prop=True,
                swap_memory=False)
            # Prepare output
            i0, O, I = returned  # i0, O, I
        if self.return_weights:
            weights = self.gather_tensors(wak='weight')
            activities = self.gather_tensors(wak='activity')
            return O, weights, activities
        else:
            return O
