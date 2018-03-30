"""2D convolutional model for Allen data."""

layer_structure = [
    {
        'layers': ['alexnet_sepgru2d'],
        'weights': [96],
        'filter_size': [7],
        'names': ['sepgru2d1'],
        'activation': ['selu'],
        'activation_target': ['post'],
    },
    {
        'layers': ['sep_conv'],
        'weights': [48],
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
        'layers': ['multi_fc'],
        'weights': [[2, 3]],
        'names': ['fc3'],
    }
]
