"""2D convolutional model for Allen data."""

layer_structure = [
    {
        # 'layers': ['alexnet_conv'],
        # 'alexnet_npy': '/media/data_cifs/vveeraba/misc/contextual_circuit_bp/alexnet_cc.npy',
        # 'alexnet_layer': 'conv1_gabors',
        'layers': ['pretrained_conv'],
        # 'alexnet_npy': '/media/data_cifs/clicktionary/pretrained_weights/cc_Gaussian_Derivatives_2d_12.npy',
        'alexnet_npy': '/media/data_cifs/clicktionary/pretrained_weights/gabors_for_contours_7.npy',
        'alexnet_layer': 's1',
        'trainable': False,
        'init_bias': True,
        # 'layers': ['conv'],
        'names': ['conv1'],
        'filter_size': [7],
        'weights': [12],
        'stride': [1, 1, 1, 1],
        'activation': ['square'],
        'activation_target': ['post'],
    },
    {
        'layers': ['pool'],
        'filter_size': [4],
        'stride': [1, 4, 4, 1],
        'names': ['pool1'],
    },
    {
        'layers': ['conv'],
        'weights': [64],
        'names': ['conv1_2'],
        'filter_size': [15],
        'stride': [1, 1, 1, 1],
        'activation': ['relu'],
        'activation_target': ['post']
    },
    {
        'layers': ['conv'],
        'weights': [2],
        'names': ['conv2'],
        'filter_size': [1],
        'stride': [1, 1, 1, 1],
        'normalization': ['batch'],
        'normalization_target': ['pre'],
    },
    {
        'layers': ['global_max_pool'],  # avg pool on left, max pool on right
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
