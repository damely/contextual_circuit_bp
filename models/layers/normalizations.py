"""Wrapper for normalization functions."""
import tensorflow as tf
from utils import py_utils
from ops.eRF_calculator import eRF_calculator
from models.layers.normalization_functions import div_norm
from models.layers.normalization_functions import layer_norm
from models.layers.normalization_functions import contextual
from models.layers.normalization_functions import contextual_psych_review
from models.layers.normalization_functions import contextual_single_ecrf
from models.layers.normalization_functions import contextual_single_ecrf_time
from models.layers.normalization_functions import contextual_single_cs


class normalizations(object):
    """Wrapper class for activation functions."""

    def __getitem__(self, name):
        """Get attribute from class."""
        return getattr(self, name)

    def __contains__(self, name):
        """Check if class contains attribute."""
        return hasattr(self, name)

    def __init__(self, kwargs=None):
        """Global variables for normalization functions."""
        self.timesteps = 3
        self.scale_CRF = True
        self.bias_CRF = True
        self.training = None
        self.strides = [1, 1, 1, 1]
        self.padding = 'SAME'
        self.update_params(kwargs)

    def update_params(self, kwargs):
        """Update the class attributes with kwargs."""
        if kwargs is not None:
            for k, v in kwargs.iteritems():
                setattr(self, k, v)

    def set_RFs(
            self,
            eRF,
            layer,
            eSRF=None,
            iSRF=None,
            SSN=None,
            SSF=None,
            V1_CRF=0.26,
            V1_neCRF=0.54,
            V1_feCRF=1.,  # 1.41,
            default_stride=1,
            rf_calculation='fit_rf',  # 'pixel_wise',
            padding=1):
        """Set RF sizes for the normalizations.

        Based on calculation of an effective RF (i.e. not
        simply kernel size of the current layer, but instead
        w.r.t. input).

        Angelucci & Shushruth 2013 V1 RFs:
        https://www.shadlenlab.columbia.edu/people/Shushruth/Lab_Page/Home_files/GalleyProof.pdf
        CRF = 0.26 degrees (1x)
        eCRF near = 0.54 degrees (2x)
        eCRF far = 1.41 degrees (5.5x). Can go as low as 1 or ~ 3.84x

        Implementation is to calculate the RF of a computational unit in
        an activity tensor. Then, near and far eCRFs are derived relative
        to the CRF size. This means that the *filter size* for the CRF is 1
        tensor pixel. And the eRFs are deduced as the appropriate filter size
        for their calculated RFs.

        For instance, units in pool_2 of VGG16 have RFs of 16x16 pixels of
        the input image. Setting the CRF filter size to 1, this means that the
        near eCRF filter size must capture an RF of ~ 32x32 pixels, and the
        far eCRF filter size must capture an RF of ~ 80x80 pixels. The eRF
        calculator can deduce these filter sizes.
        """
        if 'hardcoded_erfs' in layer.keys():
            # Use specified effective receptive fields
            self.SRF = layer['hardcoded_erfs']['SRF']
            if 'CRF_excitation' in layer['hardcoded_erfs'].keys():
                self.CRF_excitation = layer['hardcoded_erfs']['CRF_excitation']
            else:
                self.CRF_excitation = self.SRF
            if 'CRF_inhibition' in layer['hardcoded_erfs'].keys():
                self.CRF_inhibition = layer['hardcoded_erfs']['CRF_inhibition']
            else:
                self.CRF_inhibition = self.SRF
            self.SSN = layer['hardcoded_erfs']['SSN']
            self.SSF = layer['hardcoded_erfs']['SSF']
        else:
            # Calculate effective receptive field for this layer.
            # Adjust eCRF filters to yield appropriate RF sizes.
            eRFc = eRF_calculator()
            conv = {
                'padding': padding,
                'kernel': layer['filter_size'][0],
                'stride': default_stride
            }
            if len(layer['filter_size']) > 1:
                raise RuntimeError(
                    'API not implemented for layers with > 1 module.')

            # Each pixel at this layer corresponds to an input image RF
            # of eRF['r_i'].
            if rf_calculation == 'pixel_wise':
                # self.SRF = 1
                # self.CRF_excitation = 1
                # self.CRF_inhibition = 1
                self.SRF = 7
                self.CRF_excitation = 7
                self.CRF_inhibition = 7
            elif rf_calculation == 'fit_rf':
                # Adjust SRF filter, then adjust eCRFs w.r.t to that.
                self.SRF = eRFc.outFromIn(
                    conv=conv,
                    layer=eRF,
                    fix_r_out=eRF['r_i'])
                self.CRF_excitation = eRF['r_i']
                self.CRF_inhibition = eRF['r_i']
            if eSRF is not None:
                self.CRF_excitation
            if iSRF is not None:
                self.CRF_inhibition
            if SSN is None:
                SSN_eRF = py_utils.iceil(eRF['r_i'] * (V1_neCRF / V1_CRF))
                self.SSN = eRFc.outFromIn(
                    conv=conv,
                    layer=eRF,
                    fix_r_out=SSN_eRF)
            else:
                self.SSN = SSN
            if SSF is None:
                SSF_eRF = py_utils.iceil(eRF['r_i'] * (V1_feCRF / V1_CRF))
                self.SSF = eRFc.outFromIn(
                    conv=conv,
                    layer=eRF,
                    fix_r_out=SSF_eRF)
            else:
                self.SSF = SSF

    def contextual(self, x, layer, eRF, aux):
        """Contextual model from paper with frozen U & eCRFs."""
        self.update_params(aux)
        self.set_RFs(layer=layer, eRF=eRF)
        contextual_layer = contextual.ContextualCircuit(
            X=x,
            timesteps=self.timesteps,
            SRF=self.SRF,
            SSN=self.SSN,
            SSF=self.SSF,
            strides=self.strides,
            padding=self.padding,
            aux=aux)
        return contextual_layer.build()

    def contextual_single_cs(self, x, layer, eRF, aux):
        """Contextual model from paper with frozen U & eCRFs."""
        self.update_params(aux)
        self.set_RFs(layer=layer, eRF=eRF)
        contextual_layer = contextual_single_cs.ContextualCircuit(
            X=x,
            timesteps=self.timesteps,
            SRF=self.SRF,
            SSN=self.SSN,
            SSF=self.SSF,
            strides=self.strides,
            padding=self.padding,
            aux=aux)
        return contextual_layer.build()

    def contextual_single_ecrf(self, x, layer, eRF, aux):
        """Contextual model from paper with frozen U & eCRFs."""
        self.update_params(aux)
        self.set_RFs(layer=layer, eRF=eRF)
        contextual_layer = contextual_single_ecrf.ContextualCircuit(
            X=x,
            timesteps=self.timesteps,
            SRF=self.SRF,
            SSN=self.SSN,
            SSF=self.SSF,
            strides=self.strides,
            padding=self.padding,
            aux=aux)
        return contextual_layer.build()

    def contextual_single_ecrf_time(self, x, layer, eRF, aux):
        """Contextual model from paper with frozen U & eCRFs."""
        self.update_params(aux)
        self.set_RFs(layer=layer, eRF=eRF)
        contextual_layer = contextual_single_ecrf_time.ContextualCircuit(
            X=x,
            timesteps=self.timesteps,
            SRF=self.SRF,
            SSN=self.SSN,
            SSF=self.SSF,
            strides=self.strides,
            padding=self.padding,
            aux=aux)
        return contextual_layer.build()

    def contextual_psych_review(self, x, layer, eRF, aux):
        """Contextual model from paper with frozen U & eCRFs."""
        self.update_params(aux)
        self.set_RFs(layer=layer, eRF=eRF)
        contextual_layer = contextual_psych_review.ContextualCircuit(
            X=x,
            timesteps=self.timesteps,
            SRF=self.SRF,
            SSN=self.SSN,
            SSF=self.SSF,
            strides=self.strides,
            padding=self.padding,
            aux=aux)
        return contextual_layer.build()

    def divisive_2d(self, x, layer, eRF, aux):
        """Divisive normalization 2D."""
        self.update_params(aux)
        self.set_RFs(layer=layer, eRF=eRF)
        return div_norm.div_norm_2d(
            x,
            sum_window=self.CRF_excitation,
            sup_window=self.CRF_inhibition,
            gamma=self.scale_CRF,
            beta=self.bias_CRF,
            strides=self.strides,
            layer=layer,
            padding=self.padding), None, None

    def divisive_1d(self, x, layer, eRF, aux):
        """Divisive normalization 2D."""
        self.update_params(aux)
        self.set_RFs(layer=layer, eRF=eRF)
        return div_norm.div_norm_1d(
            x,
            sum_window=self.CRF_excitation,
            sup_window=self.CRF_inhibition,
            layer=layer,
            gamma=self.scale_CRF,
            beta=self.bias_CRF), None, None

    def batch(self, x, layer, eRF, aux):
        """Batch normalization."""
        self.update_params(aux)
        return tf.layers.batch_normalization(
            x,
            scale=self.scale_CRF,
            center=self.bias_CRF,
            training=self.training), None, None

    def batch_renorm(self, x, layer, eRF, aux):
        """Batch re-normalization."""
        self.update_params(aux)
        return tf.layers.batch_normalization(
            x,
            scale=self.scale_CRF,
            center=self.bias_CRF,
            training=self.training,
            renorm=True), None, None

    def layer(self, x, layer, eRF, aux):
        """Layer normalization."""
        self.update_params(aux)
        return layer_norm.layer_norm(
            x,
            gamma=self.scale_CRF,
            beta=self.bias_CRF), None, None

    def lrn(self, x, layer, eRF, aux):
        """Local response normalization."""
        self.update_params(aux)
        self.set_RFs(layer=layer, eRF=eRF)
        return tf.nn.local_response_normalization(
            x,
            depth_radius=self.CRF_inhibition,
            alpha=self.scale_CRF,
            beta=self.bias_CRF), None, None
