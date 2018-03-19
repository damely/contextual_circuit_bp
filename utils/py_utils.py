import os
import re
from datetime import datetime
import numpy as np


def get_dt_stamp():
    """Get date-timestamp."""
    return re.split(
        '\.', str(datetime.now()))[0].replace(
        ' ',
        '_').replace(
        ':',
        '_').replace(
        '-',
        '_')


def flatten_list(l, log=None):
    """Flatten a list of lists."""
    warning_msg = 'Warning: returning None.'
    assert len(l), 'Encountered empty list.'
    if l is None or l[0] is None:
        if log is not None:
            log.info(warning_msg)
        else:
            print warning_msg
        return [None]
    else:
        return [val for sublist in l for val in sublist]


def import_module(dataset, model_dir='dataset_processing'):
    """Dynamically import a module."""
    return getattr(
        __import__(model_dir, fromlist=[dataset]), dataset)


def make_dir(d):
    """Make directory d if it does not exist."""
    if not os.path.exists(d):
        os.makedirs(d)


def save_npys(data, model_name, output_string):
    """Save key/values in data as numpys."""
    for k, v in data.iteritems():
        output = os.path.join(
            output_string,
            '%s_%s' % (model_name, k)
            )
        try:
            np.save(output, v)
        except:
            print 'Failed to save %s' % k


def check_path(data_pointer, log, msg):
    """Check that the path exists."""
    if not os.path.exists(data_pointer):
        log.debug(msg)
        return False
    else:
        return data_pointer


def ifloor(x):
    """Floor as an integer."""
    return np.floor(x).astype(np.int)


def iceil(x):
    """Ceiling as an integer."""
    return np.ceil(x).astype(np.int)


def convert_to_tuple(v):
    """Convert v to a tuple."""
    if not isinstance(v, tuple):
        return tuple(v)
    else:
        return v
