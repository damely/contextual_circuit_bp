"""2D convolutional model for Allen data."""

layer_structure = [
    {
        'layers': ['conv'],
        'weights': [32],
        'names': ['conv1'],
        'filter_size': [6],
        'activation': ['selu'],
        'activation_target': ['post'],
    },
    # {
    #     'layers': ['conv'],
    #     'weights': [32],
    #     'names': ['conv2'],
    #     'filter_size': [5],
    #     'activation': ['selu'],
    #     'activation_target': ['post'],
    # },
    {
        'layers': ['pool'],
        'weights': [None],
        'names': ['pool2'],
        'filter_size': [None]
    },
    {
        'layers': ['conv'],
        'weights': [32],
        'names': ['conv3'],
        'filter_size': [6],
        'normalization': ['contextual_ff'],
        'normalization_target': ['post'],
        'normalization_aux': {
            'timesteps': 3,
            'association_field': True,
            'full_far_eCRF': True,
            'exclude_CRF': False,
            'regularization_targets': {  # Modulate sparsity
                'q_t': {
                   'regularization_type': 'orthogonal',
                   'regularization_strength': 0.01
                },
                't_t': {
                    'regularization_type': 'orthogonal',
                    'regularization_strength': 0.5
                },
                'p_t': {
                    'regularization_type': 'orthogonal',
                    'regularization_strength': 1
                },
            }
        }
    }
]

output_structure = [
    {
        'layers': ['conv'],
        'weights': [1],
        'names': ['fc4'],
        'filter_size': [1],
        'activation': ['sigmoid'],
        'activation_target': ['post']
    }
]
