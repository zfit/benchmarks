import numba as numba
import tensorflow as tf
import torch
import numpy as np
from zfit_benchmark.timer import Timer

tf.enable_v2_behavior()


def calc_np(x):
    x = x.numpy()
    x = np.sqrt(np.abs(x))
    x = np.cos(x - 0.3)
    x = np.sinh(x + 0.4)
    x  = x ** 2
    x = np.sum(np.log(x))
    return x

# @tf.function
def calc_tf(x):
    x = tf.sqrt(tf.abs(x))
    x = tf.cos(x - 0.3)
    x = tf.sinh(x + 0.4)
    x = x ** 2
    x = tf.reduce_sum(tf.log(x))
    return x


def calc_torch(x):
    x = torch.sqrt(torch.abs(x))
    x = torch.cos(x - 0.3)
    x = torch.sinh(x + 0.4)
    x = x ** 2
    x = torch.sum(torch.log(x))
    return x


@tf.function
def calc_np_wrapped(x):
    return tf.py_function(calc_np, [x], Tout=tf.float32)


@numba.jit(nopython=True)
def calc_np_numba(x):
    x = np.sqrt(np.abs(x))
    x = np.cos(x - 0.3)
    x = np.sinh(x + 0.4)
    x = x ** 2
    x = np.sum(np.log(x))
    return x


if __name__ == '__main__':
    size = (1000000,)
    x = tf.random.normal(shape=size)
    # x = torch.normal(0, 1, size=size)
    # x = x.numpy()
    with Timer() as timer:
        for _ in range(100):
            # y = calc_np_wrapped(x)
            # y = calc_np(x)
            y = calc_tf(x)
            # y = calc_torch(x)
            # y = calc_np_numba(x)
    print(y)
    print(timer.elapsed * 1000, 'ms')
