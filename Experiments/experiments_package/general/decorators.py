"""
Deccorators that are useful in certain situations.
"""
from tensorflow import random


def with_random_seed_reset(func):
    def inner():
        random.set_seed(0)
        func()

    return inner