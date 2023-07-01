"""
Deccorators that are useful in certain situations.
"""
import random

import numpy as np
import tensorflow as tf


def with_random_seed_reset(func):
    def inner(*args, **kwargs):
        tf.random.set_seed(0)
        random.seed(0)
        np.random.seed(0)
        return func(*args, **kwargs)

    return inner
