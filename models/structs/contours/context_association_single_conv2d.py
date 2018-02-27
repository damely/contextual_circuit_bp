"""2D convolutional model for Allen data."""

layer_structure = [
    {
        'layers': ['alexnet_conv'],
        'weights': [32],
        'alexnet_npy': '/media/data_cifs/vveeraba/contextual_circuit_bp/alexnet_cc.npy',
        'alexnet_layer': 'conv1_gabors',
        # 'weights': [96],
        # 'alexnet_npy': '/media/data_cifs/clicktionary/pretrained_weights/alexnet.npy',
        # 'alexnet_layer': 'conv1',
        # 'trainable': False,
        # 'xi': True,  # If FF drive is not trainable
        'names': ['conv1'],
        'filter_size': [11],
        'normalization': ['contextual_single_cs'],
        'normalization_target': ['post'],
        'normalization_aux': {
            'timesteps': 3,
            'association_field': True,
            'gru_gates': False,
            'regularization_targets': {  # Modulate sparsity
                'q_t': {
                   'regularization_type': 'l1',  # 'orthogonal',
                   'regularization_strength': 1e-7  # 0.01
                },
                'p_t': {
                    'regularization_type': 'l1',  # 'orthogonal',
                    'regularization_strength': 1e-7  # 1.
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
