"""2D convolutional model for Allen data."""

layer_structure = [
    {
        'layers': ['gru1d'],
        'weights': [48],
        'names': ['gru1'],
        'activation': ['selu'],
        'activation_target': ['post'],
    },
    {
        'layers': ['conv1d'],
        'filter_size': [3],
        'weights': [24],
        'names': ['conv2'],
        'activation': ['selu'],
        'activation_target': ['post'],
    }
]

output_structure = [
    {
        'flatten': [True],
        'flatten_target': ['pre'],
        'layers': ['fc'],
        'weights': [2],
        'names': ['fc3'],
        'activation': ['sigmoid'],
        'activation_target': ['post'],
        'squeeze': [True],
        'squeeze_target': ['post']
    }
]

