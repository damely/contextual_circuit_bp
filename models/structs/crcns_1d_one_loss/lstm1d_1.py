"""2D convolutional model for Allen data."""

layer_structure = [
    {
        'layers': ['lstm1d'],
        'weights': [16],
        'names': ['lstm1'],
        'activation': ['selu'],
        'activation_target': ['post'],
    },
    {
        'layers': ['conv1d'],
        'filter_size': [3],
        'weights': [48],
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
        'weights': [1],
        'names': ['fc3'],
        'activation': ['sigmoid'],
        'activation_target': ['post'],
        'squeeze': [True],
        'squeeze_target': ['post']
    }
]
