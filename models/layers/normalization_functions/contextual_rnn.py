import numpy as np
import tensorflow as tf
from utils import pyutils
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
            dtype=tf.float32):

        # TODO: DEDUCE SRF/SSN/SSF FROM DATA.
        self.X = X
        self.n, self.h, self.w, self.k = [int(x) for x in self.X.shape()]
        self.model_version = model_version
        self.timesteps = timesteps
        self.lesions = lesions
        self.strides = strides
        self.padding = padding
        self.SRF = SRF
        self.SSN = SSN
        self.SSF = SSF
        self.SSN_ext = 2 * pyutils.ifloor(SSN / 2.0) + 1
        self.SSF_ext = 2 * pyutils.ifloor(SSF / 2.0) + 1
        self.q_shape = [self.SRF, self.SRF, self.k, self.k]
        self.u_shape = [self.SRF, self.SRF, self.k, self.SRF]
        self.p_shape = [self.SSN_ext, self.SSN_ext, self.k, self.k]
        self.t_shape = [self.SSF_ext, self.SSF_ext, self.k, self.k]
        self.i_shape = [self.SRF, self.SRF, self.k]
        self.o_shape = [self.SRF, self.SRF, self.k]
        self.bias_shape = self.k
        self.dtype = tf.float32
        self.u_nl = tf.identity
        self.t_nl = tf.identity
        self.q_nl = tf.identity
        self.p_nl = tf.identity
        self.i_nl = tf.nn.relu  # input non linearity
        self.o_nl = tf.nn.relu  # output non linearity

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
            }
        }
        # tuned summation: pooling in h, w dimensions
        #############################################
        self[self.weight_dict['Q']['r']['weight']] = tf.get_variable(
            name=self.weight_dict['Q']['r']['weight'],
            shape=self.q_shape,
            dtype=self.dtype,
            initializer=initialization.xavier_initializer(
                            uniform=self.normal_initializer,
                            mask=None))

        # untuned suppression: reduction across feature axis
        ####################################################
        self[self.weight_dict['U']['r']['weight']] = tf.get_variable(
            name=self.weight_dict['U']['r']['weight'],
            shape=self.u_shape,
            dtype=self.dtype,
            initializer=initialization.xavier_initializer(
                            uniform=self.normal_initializer,
                            mask=None))

        # tuned summation: pooling in h, w dimensions
        #############################################
        p_array = np.zeros(self.p_shape)
        for pdx in range(self.k):
            p_array[:self.SSN, :self.SSN, pdx, pdx] = 1.0
        p_array[
            self.SSN // 2 - pyutils.ifloor(
                self.SRF / 2.0):self.SSN // 2 + pyutils.iceil(
                self.SRF / 2.0),
            self.SSN // 2 - pyutils.ifloor(
                self.SRF / 2.0):self.SSN // 2 + pyutils.iceil(
                self.SRF / 2.0),
            :,  # exclude CRF!
            :] = 0.0
        self[self.weight_dict['P']['r']['weight']] = tf.get_variable(
            name=self.weight_dict['P']['r']['weight'],
            shape=self.p_shape,
            dtype=self.dtype,
            initializer=initialization.xavier_initializer(
                            uniform=self.normal_initializer,
                            mask=p_array))

        # tuned suppression: pooling in h, w dimensions
        ###############################################
        t_array = np.zeros(self.t_shape)
        for tdx in range(self.k):
            t_array[tdx, tdx, :self.SSF, :self.SSF] = 1.0
        t_array[
            self.SSF // 2 - pyutils.ifloor(
                self.SSN / 2.0):self.SSF // 2 + pyutils.iceil(
                self.SSN / 2.0),
            self.SSF // 2 - pyutils.ifloor(
                self.SSN / 2.0):self.SSF // 2 + pyutils.iceil(
                self.SSN / 2.0),
            :,  # exclude near surround!
            :] = 0.0
        self[self.weight_dict['T']['r']['weight']] = tf.get_variable(
            name=self.weight_dict['T']['r']['weight'],
            shape=self.t_shape,
            dtype=self.dtype,
            initializer=initialization.xavier_initializer(
                            uniform=self.normal_initializer,
                            mask=t_array))

        if self.model_version != 'no_input_facing':
            # Also create input-facing weight matrices
            self[self.weight_dict['Q']['f']['weight']] = tf.get_variable(
                name=self.weight_dict['Q']['f']['weight'],
                shape=self.q_shape,
                dtype=self.dtype,
                initializer=initialization.xavier_initializer(
                                uniform=self.normal_initializer,
                                mask=None))
            self[self.weight_dict['Q']['f']['bias']] = tf.get_variable(
                name=self.weight_dict['Q']['f']['bias'],
                shape=self.bias_shape,  # Note the u_shape
                dtype=self.dtype,
                initializer=initialization.xavier_initializer(
                                uniform=self.normal_initializer,
                                mask=None))
            self[self.weight_dict['U']['f']['weight']] = tf.get_variable(
                name=self.weight_dict['U']['f']['weight'],
                shape=self.u_shape,
                dtype=self.dtype,
                initializer=initialization.xavier_initializer(
                                uniform=self.normal_initializer,
                                mask=None))
            self[self.weight_dict['U']['f']['bias']] = tf.get_variable(
                name=self.weight_dict['U']['f']['bias'],
                shape=self.bias_shape,  # Note the u_shape
                dtype=self.dtype,
                initializer=initialization.xavier_initializer(
                                uniform=self.normal_initializer,
                                mask=None))
            self[self.weight_dict['P']['f']['weight']] = tf.get_variable(
                name=self.weight_dict['P']['f']['weight'],
                shape=self.p_shape,
                dtype=self.dtype,
                initializer=initialization.xavier_initializer(
                                uniform=self.normal_initializer,
                                mask=p_array))
            self[self.weight_dict['P']['f']['bias']] = tf.get_variable(
                name=self.weight_dict['P']['f']['bias'],
                shape=self.bias_shape,  # Note the u_shape
                dtype=self.dtype,
                initializer=initialization.xavier_initializer(
                                uniform=self.normal_initializer,
                                mask=None))
            self[self.weight_dict['T']['f']['weight']] = tf.get_variable(
                name=self.weight_dict['T']['f']['weight'],
                shape=self.t_shape,
                dtype=self.dtype,
                initializer=initialization.xavier_initializer(
                                uniform=self.normal_initializer,
                                mask=t_array))
            self[self.weight_dict['T']['f']['bias']] = tf.get_variable(
                name=self.weight_dict['T']['f']['bias'],
                shape=self.bias_shape,  # Note the u_shape
                dtype=self.dtype,
                initializer=initialization.xavier_initializer(
                                uniform=self.normal_initializer,
                                mask=None))
        if self.model_version == 'full_with_cell_states':
            self[self.weight_dict['I']['r']['weight']] = tf.get_variable(
                name=self.weight_dict['I']['r']['weight'],
                shape=self.i_shape,
                dtype=self.dtype,
                initializer=initialization.xavier_initializer(
                                uniform=self.normal_initializer,
                                mask=t_array))
            self[self.weight_dict['I']['r']['bias']] = tf.get_variable(
                name=self.weight_dict['I']['r']['bias'],
                shape=self.bias_shape,
                dtype=self.dtype,
                initializer=initialization.xavier_initializer(
                                uniform=self.normal_initializer,
                                mask=None))
            self[self.weight_dict['O']['r']['weight']] = tf.get_variable(
                name=self.weight_dict['O']['r']['weight'],
                shape=self.o_shape,
                dtype=self.dtype,
                initializer=initialization.xavier_initializer(
                                uniform=self.normal_initializer,
                                mask=t_array))
            self[self.weight_dict['O']['r']['bias']] = tf.get_variable(
                name=self.weight_dict['O']['r']['bias'],
                shape=self.bias_shape,
                dtype=self.dtype,
                initializer=initialization.xavier_initializer(
                                uniform=self.normal_initializer,
                                mask=None))

        # Scalar weights
        self.alpha = tf.get_variable(shape=[], initializer=1.)
        self.tau = tf.get_variable(shape=[], initializer=1.)
        self.eta = tf.get_variable(shape=[], initializer=1.)
        self.omega = tf.get_variable(shape=[], initializer=1.)
        self.eps = tf.get_variable(shape=[], initializer=1.)
        self.tau = tf.get_variable(shape=[], initializer=1.)
        self.gamma = tf.get_variable(shape=[], initializer=1.)

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
            self[out_key] = activities

    def full(self, i0, O, I):
        """Fully parameterized contextual RNN model."""
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

        # Input
        U = self.u_nl(
            U +
            self[self.weight_dict['U']['f']['activity']] +
            self[self.weight_dict['U']['f']['bias']])
        T = self.t_nl(
            T +
            self[self.weight_dict['T']['f']['activity']] +
            self[self.weight_dict['T']['f']['bias']])
        I_summand = self.eta(self.i_nl(self.alpha * self.X - U - T))
        I = (self.eps * I) + I_summand

        # Output
        Q = self.q_nl(
            Q +
            self[self.weight_dict['Q']['f']['activity']])
        P = self.p_nl(
            P +
            self[self.weight_dict['P']['f']['activity']])
        O_summand = self.tau(self.o_nl(Q + P + (self.gamma * I)))
        O = (self.omega * O) + O_summand
        return i0, O, I

    def no_input_facing(self, i0, O, I):
        """Remove the direct FF drive to the CRF and eCRFs."""
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

        # Input
        U = self.u_nl(U + self[self.weight_dict['U']['f']['bias']])
        T = self.t_nl(T + self[self.weight_dict['T']['f']['bias']])
        I_summand = self.eta(self.i_nl(self.alpha * self.X - U - T))
        I = (self.eps * I) + I_summand

        # Output
        Q = self.q_nl(Q)
        P = self.p_nl(P)
        O_summand = self.tau(self.o_nl(Q + P + (self.gamma * I)))
        O = (self.omega * O) + O_summand
        return i0, O, I

    def no_input_scaling(self, i0, O, I):
        """Remove direct FF input to the I."""
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

        # Input
        U = self.u_nl(
            U +
            self[self.weight_dict['U']['f']['activity']] +
            self[self.weight_dict['U']['f']['bias']])
        T = self.t_nl(
            T +
            self[self.weight_dict['T']['f']['activity']] +
            self[self.weight_dict['T']['f']['bias']])
        I_summand = self.eta(self.i_nl(U - T))
        I = (self.eps * I) + I_summand

        # Output
        Q = self.q_nl(
            Q +
            self[self.weight_dict['Q']['f']['activity']])
        P = self.p_nl(
            P +
            self[self.weight_dict['P']['f']['activity']])
        O_summand = self.tau(self.o_nl(Q + P + (self.gamma * I)))
        O = (self.omega * O) + O_summand
        return i0, O, I

    def full_with_cell_states(self, i0, O, I):
        """Replace I/O scaling with weight matrices (cell states)."""
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
        I = self.conv_2d_op(
            data=I,
            weight_key=self.weight_dict['I']['r']['weight']
        ) + self[self.weight_dict['I']['r']['bias']]
        O = self.conv_2d_op(
            data=O,
            weight_key=self.weight_dict['O']['r']['weight']
        ) + self[self.weight_dict['O']['r']['bias']]

        # Input
        U = self.u_nl(
            U +
            self[self.weight_dict['U']['f']['activity']] +
            self[self.weight_dict['U']['f']['bias']])
        T = self.t_nl(
            T +
            self[self.weight_dict['T']['f']['activity']] +
            self[self.weight_dict['T']['f']['bias']])
        I_summand = self.eta(self.i_nl(self.alpha * self.X - U - T))
        I += I_summand

        # Output
        Q = self.q_nl(
            Q +
            self[self.weight_dict['Q']['f']['activity']])
        P = self.p_nl(
            P +
            self[self.weight_dict['P']['f']['activity']])
        O_summand = self.tau(self.o_nl(Q + P + (self.gamma * I)))
        O += O_summand
        return i0, O, I

    def condition(self, i0, O, I):
        """While loop halting condition."""
        return i0 < self.timesteps

    def run(self, in_array):
        """Run the backprop version of the CCircuit."""
        # Using run_reference implementation
        i0 = tf.constant(0)
        O = tf.identity(self.X)
        I = tf.identity(self.X)

        # While loop
        elems = [
            i0,
            O,
            I
        ]

        if self.model_version == 'full':
            self.conv_2d_op(
                data=self.X,
                weight_key=self.weight_dict['U']['f']['weight'],
                out_key=self.weight_dict['U']['f']['activity'])
            self.conv_2d_op(
                data=self.X,
                weight_key=self.weight_dict['T']['f']['weight'],
                out_key=self.weight_dict['T']['f']['activity'])
            self.conv_2d_op(
                data=self.X,
                weight_key=self.weight_dict['P']['f']['weight'],
                out_key=self.weight_dict['P']['f']['activity'])
            self.conv_2d_op(
                data=self.X,
                weight_key=self.weight_dict['Q']['f']['weight'],
                out_key=self.weight_dict['Q']['f']['activity'])

        returned = tf.while_loop(
            self.condition,
            self[self.model_version],
            loop_vars=elems,
            back_prop=True,
            swap_memory=False)

        # Prepare output
        return returned