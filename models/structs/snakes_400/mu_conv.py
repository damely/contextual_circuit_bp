"""2D convolutional model for Allen data."""

layer_structure = [
    {
        'layers': ['alexnet_conv'],
        'alexnet_npy': '/media/data_cifs/vveeraba/misc/contextual_circuit_bp/alexnet_cc.npy',
        'alexnet_layer': 'conv1_gabors',
        # 'layers': ['pretrained_conv'],
        # 'alexnet_npy': '/media/data_cifs/clicktionary/pretrained_weights/cc_Gaussian_Derivatives_2d_12.npy',
        # 'alexnet_npy': '/media/data_cifs/clicktionary/pretrained_weights/gabors_for_contours_7.npy',
        # 'alexnet_layer': 's1',
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
    # {
    #     'layers': ['conv'],
    #     'names': ['conv1'],
    #     'filter_size': [11],
    #     'weights': [16],
    #     'stride': [1, 1, 1, 1],
    #     'activation': ['relu'],
    #     'activation_target': ['post'],
    #     # 'normalization': ['batch'],
    #     # 'normalization_target': ['post'],
    # },
    # {
    #     'layers': ['pool'],
    #     'weights': [None],
    #     'names': ['pool1'],
    # },
    # {
    #     'layers': ['conv'],
    #     'weights': [1],
    #     'names': ['fc4'],
    #     'filter_size': [16],
    #     'stride': [1, 8, 8, 1],
    #     'activation': ['relu'],
    #     'activation_target': ['post']
    # },
    # {
    #     'layers': ['pool'],
    #     'weights': [None],
    #     'names': ['pool1'],
    #     'filter_size': [8]
    # },
    # {
    #     'layers': ['conv'],
    #     'weights': [2],
    #     'names': ['conv2'],
    #     'filter_size': [1],
    #     'stride': [1, 1, 1, 1],
    #     'activation': ['relu'],
    #     'activation_target': ['post']
    # },
    {
        'layers': ['global_pool'],
        'weights': [None],
        'names': ['pool2'],
    #     # 'activation': ['relu'],
    #     # 'activation_target': ['post']
    }
]

output_structure = [
    {
        # 'flatten': [True],
        # 'flatten_target': ['pre'],
        'layers': ['fc'],
        'weights': [2],
        'names': ['fc2'],
    }
]

