"""2D convolutional model for Allen data."""

layer_structure = [
    {
        'layers': ['alexnet_conv'],
        # 'weights': [32],
        'alexnet_npy': '/media/data_cifs/clicktionary/pretrained_weights/gabors_for_contours.npy',
        'alexnet_layer': 's1',
        # 'alexnet_npy': '/media/data_cifs/vveeraba/contextual_circuit_bp/alexnet_cc.npy',
        # 'alexnet_layer': 'conv1_gabors',
        # 'weights': [96],
        # 'alexnet_npy': '/media/data_cifs/clicktionary/pretrained_weights/alexnet.npy',
        # 'alexnet_layer': 'conv1',
        'trainable': False,
        # 'xi': True,  # If FF drive is not trainable
        'names': ['conv1'],
        'filter_size': [11],
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
    {
        'layers': ['global_pool'],
        'weights': [None],
        'names': ['pool2'],
        'activation': ['relu'],
        'activation_target': ['post']
    }
]

output_structure = [
    {
        'layers': ['fc'],
        'weights': [2],
        'names': ['fc3'],
    }
]
