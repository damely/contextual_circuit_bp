"""2D convolutional model for Allen data."""

layer_structure = [
    {
        'layers': ['sepgru2d'],
        'weights': [12],
        'filter_size': [7],
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
        'layers': ['conv'],
        'weights': [24],
        'filter_size': [3],
        'names': ['conv2'],
        'activation': ['selu'],
        'activation_target': ['post'],
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
