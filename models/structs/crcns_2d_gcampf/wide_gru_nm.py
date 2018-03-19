"""2D convolutional model for Allen data."""

layer_structure = [
    {
        'layers': ['alexnet_sepgru2d'],
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
        'layers': ['sep_conv'],
        'weights': [128],
        'filter_size': [3],
        'names': ['sepconv2'],
        'activation': ['selu'],
        'activation_target': ['post'],
        'dropout': [0.5],
        'dropout_target': ['post']
    },
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
