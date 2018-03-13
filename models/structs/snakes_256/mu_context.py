"""2D convolutional model for Allen data."""

layer_structure = [
    {
        'layers': ['alexnet_conv'],
        'alexnet_npy': '/media/data_cifs/clicktionary/pretrained_weights/gabors_for_contours_7.npy',
        'alexnet_layer': 's1',
        'trainable': False,
        'init_bias': True,
        'names': ['conv1'],
        'stride': [1, 2, 2, 1],
        'filter_size': [7],
        'hardcoded_erfs': {
            'SRF': 4,
            'CRF_excitation': 4, 
            'CRF_inhibition': 4,
            'SSN': 20,  # [11, 11, 11],
            'SSF': 20, # [11, 11, 11]  # 30
            # 'SSN': [8, 8, 8, 3],  # Vanilla VGG-style
            # 'SSF': [8, 8, 8, 3]
            # 'SSN': [6, 6, 6],  # Atrous VGG-style
            # 'SSF': [6, 6, 6]
        },
        'normalization': ['contextual_single_ecrf'],
        'normalization_target': ['post'],
        'normalization_aux': {
            'timesteps': 3,
            'xi': True,  # If FF drive is not trainable
            # 'rectify_weights': True,
            'pre_batchnorm': False,
            'post_batchnorm': False,
            'dense_connections': False,
            'atrous_convolutions': 0,
            'association_field': True,
            'multiplicative_excitation': True,
            'gru_gates': False,
            'trainable': True,
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
        'names': ['fc3'],
    }
]
