"""2D convolutional model for Allen data."""

layer_structure = [
    {
        'layers': ['alexnet_conv'],
        'weights': [32],
        'alexnet_npy': '/media/data_cifs/vveeraba/contextual_circuit_bp/alexnet_cc.npy',
        'alexnet_layer': 'conv1_gabors',
        'names': ['conv1'],
        'filter_size': [11],
        'activation': ['selu'],
        'activation_target': ['post'],
    },
    {
        'layers': ['conv'],
        'weights': [33],  # 32 Surrounds + 1x1 CRF excitation
        'names': ['conv2'],
        'filter_size': [30],
        'activation': ['selu'],
        'activation_target': ['post'],
    },
    {
        'layers': ['conv'],
        'weights': [33],  # 32 Surrounds + 1x1 CRF excitation
        'names': ['conv3'],
        'filter_size': [30],
        'activation': ['selu'],
        'activation_target': ['post'],
    }
]

output_structure = [
    {
        'layers': ['conv'],
        'weights': [1],
        'names': ['fc4'],
        'filter_size': [1],
        # 'activation': ['sigmoid'],
        # 'activation_target': ['post']
    }
]
