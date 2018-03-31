"""2D convolutional model for Allen data."""

layer_structure = [
    {
        'layers': ['_pass'],
        'names': ['contextual'],
        'hardcoded_erfs': {
            'SRF': 1,
            'CRF_excitation': 1, 
            'CRF_inhibition': 1,
            'SSN': 9,
            'SSF': 29
        },
        'normalization': ['contextual_single_ecrf_simple'],
        'normalization_target': ['pre'],
        'normalization_aux': {
            'timesteps': 10,
            'regularization_targets': {  # Modulate sparsity
                'p_t': {
                    'regularization_type': 'l1',
                    'regularization_strength': 0.8
                },
            }
        },
    }
]

output_structure = [
    {
        'layers': ['gather'],
        'aux': {
            'h': 25,
            'w': 25
        },  # Output size
        'names': ['gather'],
    },
    # {
    #     'layers': ['fc'],
    #     'weights': [1],
    #     'names': ['fc1'],
    # }
]
