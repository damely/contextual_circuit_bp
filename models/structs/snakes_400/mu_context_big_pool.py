"""2D convolutional model for Allen data."""

layer_structure = [
    {
        'layers': ['alexnet_conv'],
        # 'alexnet_npy': '/media/data_cifs/vveeraba/misc/contextual_circuit_bp/alexnet_cc.npy',
        # 'alexnet_layer': 'conv1_gabors',
        # 'layers': ['pretrained_conv'],
        # 'alexnet_npy': '/media/data_cifs/clicktionary/pretrained_weights/cc_Gaussian_Derivatives_2d_12.npy',
        'alexnet_npy': '/media/data_cifs/clicktionary/pretrained_weights/gabors_for_contours_7.npy',
        'alexnet_layer': 's1',
        'trainable': True,
        'init_bias': True,
        # 'layers': ['conv'],
        'names': ['conv1'],
        'filter_size': [7],
        'weights': [12],
        'stride': [1, 1, 1, 1],
        'activation': ['relu'],
        'activation_target': ['post'],
        'normalization': ['batch'],
        'normalization_target': ['post'],
    },
    {
        'layers': ['pool'],
        'filter_size': [2],
        'stride': [1, 2, 2, 1],
        'names': ['context1'],
        'hardcoded_erfs': {
            'SRF': 6,  # or 6
            'CRF_excitation': 6,
            'CRF_inhibition': 6,
            'SSN': 18,  # [5, 5, 5],  # or 17  # [15, 15, 15],  # [5, 5, 5],  # [11, 11, 11],
            'SSF': 18,  # [5, 5, 5],  # [15, 15, 15],  # [5, 5, 5],  # [11, 11, 11]  # 30
            # 'SSN': [5, 5, 5, 5, 5],  # Vanilla VGG-style
            # 'SSF': [5, 5, 5, 5, 5],  # Vanilla VGG-style
            # 'SSF': [8, 8, 8, 3]
            # 'SSN': [6, 6, 6],  # Atrous VGG-style
            # 'SSF': [6, 6, 6]
        },
        'normalization': ['contextual_single_ecrf_time'],
        'normalization_target': ['post'],
        'normalization_aux': {
            'timesteps': 7,
            'rectify_weights': False,  # False,
            'pre_batchnorm': False,
            'xi': True,
            'post_batchnorm': True,
            'dense_connections': False,
            'batch_norm': False,
            'atrous_convolutions': False,
            'association_field': True,
            'multiplicative_excitation': False,
            'gru_gates': False,
            'trainable': True,
            'regularization_targets': {  # Modulate sparsity
                'q_t': {
                   'regularization_type': 'laplace',  # 'orthogonal',
                   'regularization_strength': 1e-3  # 1e-5  # 0.01
                },
                'p_t': {
                    'regularization_type': 'laplace',  # 'laplace',  # 'orthogonal',
                    'regularization_strength': 1e-3  # 1e-5  # 1.
                },
            }
        }
    },
    {
        'layers': ['conv'],
        'weights': [2],
        'names': ['conv2'],
        'filter_size': [1],
        'stride': [1, 1, 1, 1],
        'activation': ['relu'],
        'activation_target': ['post'],
        'normalization': ['batch'],
        'normalization_target': ['post'],
    },
    {
        'layers': ['global_pool'],  # avg pool on left, max pool on right
        'weights': [None],
        'names': ['pool2'],
    #     # 'activation': ['relu'],
    #     # 'activation_target': ['post']
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
