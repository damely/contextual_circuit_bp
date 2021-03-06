"""2D convolutional model for Allen data."""

layer_structure = [
    {
        'layers': ['pretrained_conv'],
        # 'alexnet_npy': '/media/data_cifs/clicktionary/pretrained_weights/gabors_for_contours_11.npy',
        # 'alexnet_layer': 's1',
        'alexnet_npy': '/media/data_cifs/vveeraba/misc/contextual_circuit_bp/alexnet_cc.npy',
        'alexnet_layer': 'conv1_gabors',
        # 'nonlinearity': 'square',
        'trainable': True,
        'init_bias': True,
        # 'layers': ['conv'],
        'names': ['conv1'],
        'stride': [1, 1, 1, 1],
        'weights': [32],
        'filter_size': [11],
        'normalization': ['batch'],
        'normalization_target': ['post'],
    },
    # {
    #     'layers': ['conv'],
    #     'names': ['conv1'],
    #     'weights': [8],
    #     'filter_size': [5],
    # },
    {
        'layers': ['_pass'],
        'names': ['context1'],
        'hardcoded_erfs': {
            'SRF': 8,
            'CRF_excitation': 8,
            'CRF_inhibition': 8,
            'SSN': [11, 11, 11],  # [5, 5, 5],  # [11, 11, 11],
            'SSF': [11, 11, 11],  # [5, 5, 5],  # [11, 11, 11]  # 30
            # 'SSN': [5, 5, 5, 5, 5],  # Vanilla VGG-style
            # 'SSF': [5, 5, 5, 5, 5],  # Vanilla VGG-style
            # 'SSF': [8, 8, 8, 3]
            # 'SSN': [6, 6, 6],  # Atrous VGG-style
            # 'SSF': [6, 6, 6]
        },
        'normalization': ['contextual_single_ecrf_time'],
	'normalization_target': ['post'],
	'normalization_aux': {
            'timesteps': 5,
            'rectify_weights': False,  # False,
            'pre_batchnorm': False,
            'post_batchnorm': False,
            'dense_connections': False,
            # 'batch_norm': True,
            'atrous_convolutions': False,
            'association_field': True,
            'multiplicative_excitation': False,
            'gru_gates': False,
            'trainable': True,
            'regularization_targets': {  # Modulate sparsity
                'q_t': {
                   'regularization_type': 'l1',  # 'orthogonal',
                   'regularization_strength': 1e-7  # 1e-5  # 0.01
                },
                'p_t': {
                    'regularization_type': 'l1',  # 'laplace',  # 'orthogonal',
                    'regularization_strength': 1e-7  # 1e-5  # 1.
                },
            }
        }
    },
    {
        'layers': ['global_max_pool'],
        'weights': [None],
        'names': ['pool2'],
    #     'normalization': ['batch'],
    #     'normalization_target': ['post'],
    #     'activation': ['relu'],
    #     'activation_target': ['post']
    }
]

output_structure = [
    {
        'flatten': [True],
        'flatten_target': ['pre'],
        'layers': ['fc'],
        'weights': [2],
        'names': ['fc2'],
    }
]

