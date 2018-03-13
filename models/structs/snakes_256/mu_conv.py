"""2D convolutional model for Allen data."""

layer_structure = [
    {
        'layers': ['alexnet_conv'],
        'alexnet_npy': '/media/data_cifs/clicktionary/pretrained_weights/gabors_for_contours_7.npy',
        'alexnet_layer': 's1',
        'trainable': False,
        'init_bias': True,
        'names': ['conv1'],
        'filter_size': [7],
        'stride': [1, 2, 2, 1],
        'activation': ['relu'],
        'activation_target': ['post'],
    },
    # {
    #     'layers': ['pool'],
    #     'weights': [None],
    #     'names': ['pool1'],
    #     'filter_size': [8]
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
    #     'layers': ['global_pool'],
    #     'weights': [None],
    #     'names': ['pool2'],
        # 'activation': ['relu'],
        # 'activation_target': ['post']
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
