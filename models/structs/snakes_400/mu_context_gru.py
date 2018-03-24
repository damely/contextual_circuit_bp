"""2D convolutional model for Allen data."""

layer_structure = [
    {
        'layers': ['pretrained_conv'],
        'alexnet_npy': '/media/data_cifs/clicktionary/pretrained_weights/gabors_for_contours_11.npy',
        'alexnet_layer': 's1',
        'nonlinearity': 'square',
        'trainable': False,
        'init_bias': True,
        # 'layers': ['conv'],
        'names': ['conv1'],
        'stride': [1, 2, 2, 1],
        'weights': [8],
        'filter_size': [11],
        'hardcoded_erfs': {
            'SRF': 6,
            'CRF_excitation': 6,
            'CRF_inhibition': 6,
            'SSN': [29],  # [5, 5, 5],  # [11, 11, 11],
            'SSF': [29],  # [5, 5, 5],  # [11, 11, 11]  # 30
            # 'SSN': [5, 5, 5, 5, 5],  # Vanilla VGG-style
            # 'SSF': [5, 5, 5, 5, 5],  # Vanilla VGG-style
            # 'SSF': [8, 8, 8, 3]
            # 'SSN': [6, 6, 6],  # Atrous VGG-style
            # 'SSF': [6, 6, 6]
        },
        'normalization': ['contextual_single_ecrf'],
        'normalization_target': ['post'],
        'normalization_aux': {
            'timesteps': 5,
            'rectify_weights': False,
            'pre_batchnorm': True,
            'post_batchnorm': False,
            # 'dense_connections': True,
            # 'batch_norm': True,
            'atrous_convolutions': False,
            'association_field': True,
            'multiplicative_excitation': True,
            'gru_gates': True,
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
    # {
    #     'layers': ['global_pool'],
    #     'weights': [None],
    #     'names': ['pool2'],
    #     # 'activation': ['relu'],
    #     # 'activation_target': ['post']
    # }
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
