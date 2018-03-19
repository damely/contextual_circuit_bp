"""Pull summaries from an experiment and load up their tensorboard."""
import os
import re
import subprocess
import argparse
import commands
import numpy as np
from db import db


def get_process_id(name):
    """Return process ids found by (partial) name or regex.

    >>> get_process_id('kthreadd')
    [2]
    >>> get_process_id('watchdog')
    [10, 11, 16, 21, 26, 31, 36, 41, 46, 51, 56, 61]  # ymmv
    >>> get_process_id('non-existent process')
    []
    """
    child = subprocess.Popen(
        ['pgrep', '-f', name],
        stdout=subprocess.PIPE,
        shell=False)
    response = child.communicate()[0]
    return [int(pid) for pid in response.split()]


def list_ports(ports, min_port, max_port, possible_ports):
    """Build up port list."""
    port_list = []
    for m in range(min_port, max_port):
        for p in possible_ports:
            port_num = int('%s00%s' % (m, p))
            if port_num not in ports:
                port_list += [port_num]
    return port_list


def main(
        experiment_name,
        kill_tensorboards,
        min_port=6,
        max_port=9,
        possible_ports=[6, 7, 8, 9],
        selected_port=None):
    """Pull summaries and plot them in a tensorboard."""
    summaries = db.get_summary_list(experiment_name)
    pids = get_process_id('tensorboard')
    if kill_tensorboards:
        [os.kill(pid, 0) for pid in pids]
        print 'Killed tensorboards with pids: %s' % pids

    # Look up ports for pids
    if selected_port is None:
        ports = []
        for pid in pids:
            stdout = commands.getstatusoutput(
                'ss -l -p -n | grep '",%s,"'' % pid)
            reg_search = re.search(r'(?<=:)\d+', str(stdout))
            if reg_search is not None:
                ports += [int(reg_search.group())]
            else:
                ports += [False]
        available_ports = np.asarray(
            list_ports(
                ports=ports,
                min_port=min_port,
                max_port=max_port,
                possible_ports=possible_ports))
        selected_port = available_ports[0]

    # Make summary string
    log_str = ''
    for idx, summary in enumerate(summaries):
        name = summary['model_struct'].split('/')[1]
        print 'Model %s: %s' % (idx, name)
        path = summary['summary_dir']
        if idx != len(summaries) - 1:
            log_str += '%s:%s,' % (name, path)
        else:
            log_str += '%s:%s' % (name, path)

    # Launch tensorboard
    os.system(
        'tensorboard --port=%s --logdir=%s' % (
            selected_port,
            log_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--experiment',
        dest='experiment_name',
        type=str,
        default=None,
        help='Name of the experiment.')
    parser.add_argument(
        '--port',
        dest='selected_port',
        type=int,
        default=None,
        help='Port to push tensorboard to.')
    parser.add_argument(
        '--kill',
        dest='kill_tensorboards',
        action='store_true',
        help='Kill all running tensorboards.')
    args = parser.parse_args()
    main(**vars(args))
