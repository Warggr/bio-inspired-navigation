''' This code has been adapted from:
***************************************************************************************
*    Title: "Scaling Local Control to Large Scale Topological Navigation"
*    Author: "Xiangyun Meng, Nathan Ratliff, Yu Xiang and Dieter Fox"
*    Date: 2020
*    Availability: https://github.com/xymeng/rmp_nav
*
***************************************************************************************
'''
import os
import glob
import tabulate
from typing import Any


def pprint_dict(x):
    """
    :param x: a dict
    :return: a string of pretty representation of the dict
    """

    def helper(d):
        ret = {}
        for k, v in d.items():
            if isinstance(v, dict):
                ret[k] = helper(v)
            else:
                ret[k] = v
        return tabulate.tabulate(ret.items())

    return helper(x)


def str_to_dict(s, delim=',', kv_delim='='):
    ss = s.split(delim)
    d = {}
    for s in ss:
        if s == '':
            continue
        field, value = s.split(kv_delim)
        try:
            value = eval(value, {'__builtins__': None})
        except:
            # Cannot convert the value. Treat it as it is.
            pass
        d[field] = value
    return d


def module_grad_stats(module):
    headers = ['layer', 'max', 'min']

    def maybe_max(x):
        return x.max() if x is not None else 'None'

    def maybe_min(x):
        return x.min() if x is not None else 'None'

    data = [
        (name, maybe_max(param.grad), maybe_min(param.grad))
        for name, param in module.named_parameters()
    ]
    return tabulate.tabulate(data, headers, tablefmt='psql')


def module_weights_stats(module):
    headers = ['layer', 'max', 'min']

    def maybe_max(x):
        return x.max() if x is not None else 'None'

    def maybe_min(x):
        return x.min() if x is not None else 'None'

    data = [
        (name, maybe_max(param), maybe_min(param))
        for name, param in module.named_parameters()
    ]
    return tabulate.tabulate(data, headers, tablefmt='psql')

DATA_STORAGE_FOLDER = os.path.join(os.path.dirname(__file__), "..", "data", "models")

def load_model(filepath, step=None, load_to_cpu=False) -> tuple[dict[str, Any], int]:
    '''
    :param filepath: The path to the model file, without .{epoch} extension
    :param step: if None. Load the latest.
    :return: a tuple (state, epoch) with `epoch` the latest epoch found and `saved` the saved state dict at that epoch
    '''
    import torch
    if not step:
        files = glob.glob(filepath + '.*')
        parsed = []
        for fn in files:
            try:
                r = int(fn.split('.')[-1])
            except Exception:
                print('Ignoring file %s', fn)
                continue
            parsed.append(r)

        if not parsed:
            raise FileNotFoundError(f"Could not find any iteration of {os.path.basename(filepath)} in {os.path.dirname(filepath)}")
        step = max(parsed)

    path = filepath + '.' + str(step)

    if not os.path.isfile(path):
        raise ValueError(path + ' is not a valid path')
    if load_to_cpu:
        return torch.load(path, map_location=lambda storage, location: storage), step
    else:
        return torch.load(path), step
