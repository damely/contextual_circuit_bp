"""2D convolutional model for Allen data."""

layer_structure = [
    {
        'layers': ['alexnet_sgru2d'],
        'weights': [96],
        'alexnet_npy': '/media/data_cifs/clicktionary/pretrained_weights/alexnet.npy',
        'alexnet_layer': 'conv1',
        'trainable': True,
        'init_bias': True,
        'cam_mask': False,
        'filter_size': [11],
        'names': ['sepgru2d1'],
        'activation': ['selu'],
        'activation_target': ['post'],
    },
    {
        'layers': ['pool'],
        'weights': [None],
        'names': ['pool2'],
        'filter_size': [None]
    },
    {
        'layers': ['fc'],
        'weights': [128],
        'names': ['fc1'],
        'flatten': [True],
        'flatten_target': ['pre'],
        'dropout': [0.5],
        'dropout_target': ['post'],
        'activation': ['selu'],
        'activation_target': ['post'],
        'regularization_type': ['l2'],
        'regularization_target': ['post'],
        'regularization_strength': [5e-7]
    }
]

output_structure = [
    {
        'layers': ['fc'],
        'weights': [1],  # Output size
        'names': ['fc2'],
        'regularization_type': ['l2'],
        'regularization_target': ['post'],
        'regularization_strength': [5e-7]
    }
]
