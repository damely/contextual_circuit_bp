
def plotly_credentials():
    return {
        'username': 'linsleyd',
        'api_key': 'JbcKSw6nXngS4FhQzelb'
    }


def postgresql_credentials():
    return {
            'username': 'contextual_DCN_p3',
            'password': 'serrelab'
           }


def postgresql_connection(port=''):
    unpw = postgresql_credentials()
    params = {
        'database': 'contextual_DCN_p3',
        'user': unpw['username'],
        'password': unpw['password'],
        'host': 'localhost',
        'port': port,
    }
    # params = {
    #     'database': 'mnist',
    #     'user': 'andrew',
    #     'password': 'serrelab',
    #     'host': 'localhost',
    #     'port': port,
    # }
    return params


def machine_credentials():
    """Credentials for your machine."""
    return {
        'username': 'drew',
        'password': 'serrelab',
        'ssh_address': 'serrep3.services.brown.edu'
       }


# def cluster_credentials():
#     return {
#         'username': 'drew',
#         'password': 'serrelab',
#         'ssh_address': 'serrep3.services.brown.edu'
#        }

