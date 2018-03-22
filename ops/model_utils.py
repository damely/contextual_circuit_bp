"""Build DNN models from python dictionaries."""
import json
import numpy as np
import tensorflow as tf
# from models.layers import ff
# from models.layers import pool
from models.layers.ff import ff
from models.layers.activations import activations
from models.layers.normalizations import normalizations
from models.layers.regularizations import regularizations
from ops.eRF_calculator import eRF_calculator


eRF = eRF_calculator()


class model_class(object):
    """Default model class that is generated with a layer_structure dict."""

    def __getitem__(self, name):
        """Get class attributes."""
        return getattr(self, name)

    def __contains__(self, name):
        """Check class attributes."""
        return hasattr(self, name)

    def __init__(self, mean, training, output_size, **kwargs):
        """Set model to trainable/not and pass the mean values."""
        self.var_dict = {}
        self.data_dict = None
        self.regularizations = {}
        self.training = training
        self.mean = mean
        self.output_size = output_size
        self.share_vars = ['training', 'output_size']
        self.layer_vars = {k: self[k] for k in self.share_vars}
        input_normalization = kwargs.get('input_normalization', 'mean')
        if input_normalization is None:
            input_normalization = 'mean'
        self.input_normalization = input_normalization

    def build(
            self,
            data,
            layer_structure,
            output_structure=None,
            output_size=None,
            log=None,
            tower_name='cnn'):
        """Main model creation method."""
        if self.mean is not None and self.input_normalization != 'none':
            if isinstance(self.mean, dict):
                if self.input_normalization == 'mean':
                    data -= self.mean['mean']
                elif self.input_normalization == 'max':
                    data /= (
                        np.expand_dims(
                            self.mean['max'], axis=0)).astype(np.float32)
                elif self.input_normalization == 'zscore':
                    data -= self.mean['mean']
                    data /= self.mean['std']
                else:
                    raise NotImplementedError
            else:
                # If there's a mean shape mismatch default to channel means
                test_shape = [int(x) for x in data.get_shape()]
                if np.any(
                    np.asarray(
                        self.mean.shape[:2]) != np.asarray(
                        test_shape[1:3])):
                    self.mean = np.mean(np.mean(self.mean, axis=0), axis=0)
                    log.info(
                        'Mismatch between saved mean and input data. ' +
                        'Using channel means instead.')
                if len(self.mean.shape) == 4:
                    # For ST models, combine across time
                    data -= self.mean.mean(0)
                else:
                    if not np.all(
                            self.mean.shape == data.get_shape().as_list()):
                        log.info(
                            'Reducing mean to scalar because of ' +
                            'dimension mismatch.')
                        data -= self.mean.mean()
                    else:
                        data -= (np.expand_dims(
                            self.mean, axis=0)).astype(np.float32)
        else:
            log.debug('Empty mean tensor. No mean adjustment.')
            # log.debug('Failed to mean-center data.')
        input_data = tf.identity(data, name='lrp_input')
        assert log is not None, 'You must pass a logger.'

        # Calculate eRF info for main tower
        tower_eRFs = eRF.calculate(
            layer_structure,
            data,
            verbose=True,
            log=log)
        log.info('Creating main model tower.')
        self, features, layer_summary = create_conv_tower(
            self=self,
            act=input_data,
            layer_structure=layer_structure,
            tower_name=tower_name,
            eRFs=tower_eRFs,
            layer_summary=None)

        # Add output layer if one is not specified
        if output_structure is None:
            assert self.output_size is not None, 'Give model an output shape.'
            output_structure = self.default_output_layer()

        # Calculate output effective receptive fields
        if tower_eRFs is not None:
            output_eRFs = eRF.calculate(
                output_structure,
                features,
                r_i=tower_eRFs.items()[-1][1],
                verbose=True)
        else:
            output_eRFs = [None] * len(output_structure)
            log.debug('Could not calculate effective RFs of units.')

        # Build the output tower.
        log.info('Creating output tower.')
        self, output, layer_summary = create_conv_tower(
            self=self,
            act=features,
            layer_structure=output_structure,
            tower_name='output',
            eRFs=output_eRFs,
            layer_summary=layer_summary)
        if isinstance(output, list):
            self.output = output
        else:
            self.output = tf.identity(output, name='output')
        self.data_dict = None
        return output, layer_summary

    def default_output_layer(self):
        """Apply a default output layer if none is specified."""
        return [
            {
                'layers': ['fc'],
                'names': ['output'],
                'flatten': [True],
                'flatten_target': ['pre'],
                'filter_size': [1],
                'weights': [self.output_size]
            }
        ]

    def save_npy(self, sess, npy_path="./saved_weights.npy"):
        """Default method: Save your model's weights to a numpy."""
        assert isinstance(sess, tf.Session)

        data_dict = {}
        num_files = 0

        for (name, idx), var in self.var_dict.items():
            # print(var.get_shape())
            var_out = sess.run(var)
            if name not in data_dict.keys():
                data_dict[name] = {}
            data_dict[name][idx] = var_out
        np.save('%s%s' % (npy_path, str(num_files)), data_dict)
        print 'Weights saved to: %s' % npy_path
        return npy_path

    def get_var_count(self):
        """Default method: Count your variables."""
        count = 0
        for v in self.var_dict.values():
            count += reduce(lambda x, y: x * y, v.get_shape().as_list())
        return count


def update_summary(layer_summary, op_name):
    """Produce a summary of the structure of a CNN used in an experiment."""
    if layer_summary is None:
        bottom_name = 'Input'
        layer_summary = []
    else:
        bottom_name = layer_summary[len(layer_summary) - 1].split(
            'Operation: ')[-1].strip('\n')
    assert isinstance(op_name, basestring), 'Pass a string to update_summary.'
    layer_summary += ['Bottom: %s | Operation: %s\n' % (
        bottom_name, op_name)]
    return layer_summary


def flatten_op(self, it_dict, act, layer_summary, eRFs, target):
    """Wrapper for a flatten operation in a graph."""
    if 'flatten_target' in it_dict.keys() and \
            it_dict['flatten_target'][0] == target:
        tshape = [int(x) for x in act.get_shape()]
        rows = tshape[0]
        cols = np.prod(tshape[1:])
        act = tf.reshape(act, [rows, cols])
        layer_summary = update_summary(
            layer_summary=layer_summary,
            op_name='flattened')
    return self, act, layer_summary


def squeeze_op(self, it_dict, act, layer_summary, eRFs, target):
    """Wrapper for a squeeze operation in a graph."""
    if 'squeeze_target' in it_dict.keys() and \
            it_dict['squeeze_target'][0] == target:
        act = tf.squeeze(act)
        layer_summary = update_summary(
            layer_summary=layer_summary,
            op_name='squeezed')
    return self, act, layer_summary


def dropout_op(self, it_dict, act, layer_summary, reg_mod, eRFs, target):
    """Wrapper for a dropout operation in a graph."""
    if 'dropout_target' in it_dict.keys() and \
            it_dict['dropout_target'][0] == target:
        dropout_prop = it_dict['dropout'][0]
        act = reg_mod.dropout(act, keep_prob=dropout_prop)
        layer_summary = update_summary(
            layer_summary=layer_summary,
            op_name='dropout_%s' % dropout_prop)
    return self, act, layer_summary


def activ_op(self, it_dict, act, layer_summary, activ_mod, eRFs, target):
    """Wrapper for an activation operation in a graph."""
    if 'activation_target' in it_dict.keys() and \
            it_dict['activation_target'][0] == target:
        activation = it_dict['activation'][0]
        act = activ_mod[activation](act)
        layer_summary = update_summary(
            layer_summary=layer_summary,
            op_name=activation)
    return self, act, layer_summary


def norm_op(self, it_dict, act, layer_summary, norm_mod, eRFs, target):
    """Wrapper for a normalization operation in a graph."""
    if 'normalization_target' in it_dict.keys() and \
            it_dict['normalization_target'][0] == target:
        normalization = it_dict['normalization'][0]
        if 'normalization_aux' in it_dict:
            aux = it_dict['normalization_aux']
        else:
            aux = None
        if len(it_dict['names']) > 1:
            raise RuntimeError('TODO: Fix implementation for multiple names.')
        if eRFs is None:
            # Use hardcoded eRF sizes
            layer_eRFs = None
        else:
            layer_eRFs = eRFs[it_dict['names'][0]]
        act, weights, activities = norm_mod[normalization](
            act,
            layer=it_dict,
            eRF=layer_eRFs,
            aux=aux)
        if weights is not None:
            self = attach_weights(
                self,
                weights,
                layer_name=it_dict['names'][0])
            # TODO: Attach activities too.
        self = attach_regularizations(
            self,
            weights,
            activities,
            aux,
            layer_name=it_dict['names'][0])
        layer_summary = update_summary(
            layer_summary=layer_summary,
            op_name=normalization)
    return self, act, layer_summary


def ff_op(self, it_dict, act, layer_summary, ff_mod):
    """Wrapper for a FF operation in a graph."""
    fla = it_dict.get('layers', [None])
    fns = it_dict.get('names', [None])
    fws = it_dict.get('weights', [None])
    ffs = it_dict.get('filter_size', [1, 2, 2, 1])  # default filter size
    for ff_attr, ff_name, ff_wght, ff_fs in zip(fla, fns, fws, ffs):
        in_channels = int(act.get_shape()[-1])
        if hasattr(ff_mod, ff_attr):
            self, act = ff_mod[ff_attr](
                context=self,
                act=act,
                in_channels=in_channels,
                out_channels=ff_wght,
                filter_size=ff_fs,
                it_dict=it_dict,
                name=ff_name)
            layer_summary = update_summary(
                layer_summary=layer_summary,
                op_name=ff_attr)
        else:
            print '*' * 60
            print 'Operation %s is not yet implemented.' % ff_attr
            print '*' * 60
            raise NotImplementedError

    return self, act, layer_summary


def attach_weights(self, weights, layer_name):
    """Attach weights to model."""
    for k, v in weights.iteritems():
        if '_b' in k:
            w_or_b = 1
        else:
            w_or_b = 0
        it_key = ('%s_%s' % (layer_name, k), w_or_b)
        self.var_dict[it_key] = v
    return self


def attach_regularizations(
        self,
        weights,
        activities,
        aux,
        layer_name,
        it_dict=None,
        side=None,
        a_or_w='weights'):
    """Attach regularizations. TODO combine this and other reg. interface."""
    if it_dict is not None and 'regularization_type' in it_dict:
        # Regularization of model layers
        target = it_dict['regularization_target'][0]
        wd_type = it_dict['regularization_type'][0]
        if target == side and wd_type is not None:
            reg_strength = it_dict['regularization_strength'][0]
            if 'regularization_activities_or_weights' in it_dict.keys():
                a_or_w = it_dict['regularization_activities_or_weights']
            if a_or_w == 'weights':
                weights = self.var_dict[('%s' % layer_name, 0)]
            else:
                weights = activities
            it_key = '%s_%s_%s' % (layer_name, side, a_or_w)
            self.regularizations[it_key] = {
                'weight': weights,
                'regularization_type': wd_type,
                'regularization_strength': reg_strength
            }
    if aux is not None and (
        'regularization_type' in aux.keys() or
            'regularization_targets' in aux.keys()):
        # Auxillary regularizations
        if 'regularization_targets' in aux.keys():
            # Target specific weights
            filter_dict = aux['regularization_targets']
            new_weights, wd_type_dict, reg_strength_dict = {}, {}, {}
            for k, v in filter_dict.iteritems():
                new_weights[k] = weights[k]
                wd_type_dict[k] = v['regularization_type']
                reg_strength_dict[k] = v['regularization_strength']
            weights = new_weights
        else:
            # Uniformly apply across weights
            wd_type = aux['regularization_type']
            reg_strength = aux['regularization_strength']
            if 'regularization_activities_or_weights' in aux.keys():
                a_or_w = aux['regularization_activities_or_weights']
            if a_or_w == 'weights':
                pass
            else:
                weights = activities
            wd_type_dict, reg_strength_dict = {}, {}
            for k, v in weights.keys():
                wd_type_dict[k] = wd_type
                reg_strength_dict[k] = reg_strength
        for k, v in weights.iteritems():
            it_key = '%s_%s' % (layer_name, k)
            self.regularizations[it_key] = {
                'weight': v,
                'regularization_type': wd_type_dict[k],
                'regularization_strength': reg_strength_dict[k]
            }
        print 'Adding layer regularizers:'
        print json.dumps(self.regularizations.keys(), indent=4)
    return self


def create_conv_tower(
        self,
        act,
        layer_structure,
        tower_name,
        eRFs=None,
        layer_summary=None):
    """
    Construct a feedforward neural model tower.
    Inputs:::
    act: a tensor to be fed into the model.
    layer_structure: a list of dictionaries that
    specify model layers. Note that not all operations are commutative
    (e.g. act fun -> dropout -> normalization).
    tower_name: name of the tower's variable scope.
    """
    activ_mod = activations(self.layer_vars)
    norm_mod = normalizations(self.layer_vars)
    reg_mod = regularizations(self.layer_vars)
    ff_mod = ff(self.layer_vars)
    with tf.variable_scope(tower_name):
        for it_dict in layer_structure:
            it_name = it_dict['names'][0]
            # it_neuron_op = it_dict['layers'][0]
            self, act, layer_summary = squeeze_op(
                self,
                it_dict,
                act,
                layer_summary,
                eRFs=eRFs,
                target='pre')
            self, act, layer_summary = flatten_op(
                self,
                it_dict,
                act,
                layer_summary,
                eRFs=eRFs,
                target='pre')
            self = attach_regularizations(
                self,
                weights=None,
                activities=act,
                aux=None,
                layer_name=it_name,
                it_dict=it_dict,
                side='pre')
            self, act, layer_summary = dropout_op(
                self,
                it_dict,
                act,
                layer_summary,
                reg_mod,
                eRFs=eRFs,
                target='pre')
            self, act, layer_summary = activ_op(
                self,
                it_dict,
                act,
                layer_summary,
                activ_mod,
                eRFs=eRFs,
                target='pre')
            self, act, layer_summary = norm_op(
                self,
                it_dict,
                act,
                layer_summary,
                norm_mod,
                eRFs=eRFs,
                target='pre')
            # self, act = ff.pool_ff_interpreter(
            #     self=self,
            #     it_neuron_op=it_neuron_op,
            #     act=act,
            #     it_name=it_name,
            #     it_dict=it_dict)
            # Filter and pool operations
            self, act, layer_summary = ff_op(
                self=self,
                act=act,
                it_dict=it_dict,
                layer_summary=layer_summary,
                ff_mod=ff_mod)
            # layer_summary = update_summary(
            #     layer_summary=layer_summary,
            #     op_name=it_dict['layers'])
            self, act, layer_summary = norm_op(
                self,
                it_dict,
                act,
                layer_summary,
                norm_mod,
                eRFs=eRFs,
                target='post')
            self, act, layer_summary = activ_op(
                self,
                it_dict,
                act,
                layer_summary,
                activ_mod,
                eRFs=eRFs,
                target='post')
            self, act, layer_summary = dropout_op(
                self,
                it_dict,
                act,
                layer_summary,
                reg_mod,
                eRFs=eRFs,
                target='post')
            self = attach_regularizations(
                self,
                weights=None,
                activities=act,
                aux=None,
                layer_name=it_name,
                it_dict=it_dict,
                side='post')
            self, act, layer_summary = flatten_op(
                self,
                it_dict,
                act,
                layer_summary,
                eRFs=eRFs,
                target='post')
            self, act, layer_summary = squeeze_op(
                self,
                it_dict,
                act,
                layer_summary,
                eRFs=eRFs,
                target='post')
            setattr(self, it_name, act)
            print 'Added layer: %s' % it_name
    return self, act, layer_summary
