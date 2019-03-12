import numpy as np
import tensorflow as tf

import random

def minibatch(*array, **kwargs):

    batch_size = kwargs.get('batch_size', 128)

    if len(array) == 1:
        array = array[0]
        for i in range(0, len(array), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(array[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in array)


def shuffle(*arrays, **kwargs):

    require_indices = kwargs.get('indices', False)

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('All inputs to shuffle must have '
                         'the same length.')

    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)

    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)

    if require_indices:
        return result, shuffle_indices
    else:
        return result


def assert_no_grad(variable):

    if variable.requires_grad:
        raise ValueError(
            "nn criterions don't compute the gradient w.r.t. targets - please "
            "mark these variables as volatile or not requiring gradients"
        )


def set_seed(seed, cuda=False):

    np.random.seed(seed)
    random.seed(seed)


def str2bool(v):
    return v.lower() in ('true')