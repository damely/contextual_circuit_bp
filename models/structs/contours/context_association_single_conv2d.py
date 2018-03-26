"""2D convolutional model for Allen data."""

layer_structure = [
    {
        'layers': ['alexnet_conv'],
        'weights': [32],
        'alexnet_npy': '/media/data_cifs/vveeraba/misc/contextual_circuit_bp/alexnet_cc.npy',
        'alexnet_layer': 'conv1_gabors',
        # 'weights': [96],
        # 'alexnet_npy': '/media/data_cifs/clicktionary/pretrained_weights/alexnet.npy',
        # 'alexnet_layer': 'conv1',
        # 'trainable': False,
        # 'xi': True,  # If FF drive is not trainable
        'names': ['conv1'],
        'filter_size': [11],
        'hardcoded_erfs': {
            'SRF': 6,
            'CRF_excitation': 6, 
            'CRF_inhibition': 6,
            'SSN': 24,
            'SSF': 24
            # 'SSN': [8, 8, 8, 3],  # Vanilla VGG-style
            # 'SSF': [8, 8, 8, 3]
            # 'SSN': [6, 6, 6],  # Atrous VGG-style
            # 'SSF': [6, 6, 6]
        },
        # 'hardcoded_erfs': {
        #     'SRF': 11,
        #     'CRF_excitation': 6, 
        #     'CRF_inhibition': 6,
        #     'SSN': 30,
        #     'SSF': 30
        # },
        'normalization': ['old_cc'],  # ['contextual_single_ecrf_time'],
        'normalization_target': ['post'],
        'normalization_aux': {
            'timesteps': 7,
            'pre_batchnorm': False,
            'post_batchnorm': False,
            'dense_connections': True,
            'atrous_convolutions': 0,
            'association_field': True,
            'gru_gates': False,
            'regularization_targets': {  # Modulate sparsity
                'q_t': {
                   'regularization_type': 'l1',  # 'orthogonal',
                   'regularization_strength': 1e-7  # 0.01
                },
                'p_t': {
                    'regularization_type': 'l1',  # 'laplace',  # 'orthogonal',
                    'regularization_strength': 1e-7  # 1.
                    },
            }
        }
    }
]

output_structure = [
    {
        'layers': ['conv'],
        'weights': [1],
        'names': ['fc4'],
        'filter_size': [1],
        'activation': ['sigmoid'],
        'activation_target': ['post']
    }
]
