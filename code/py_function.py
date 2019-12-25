import numba as numba
import tensorflow as tf
import torch
import numpy as np
from zfit_benchmark.timer import Timer

# v2behavior = True
v2behavior = False
if v2behavior:
    tf.enable_v2_behavior()
import zfit

if not v2behavior:
    tf.disable_eager_execution()

global_y = tf.random.normal(shape=(10, 1))
var1 = tf.Variable(42.)

list1 = [1, 2, 3, 4]


def dummy():
    normal = np.random.normal(size=100000)
    list1.append(np.sum(normal))


def calc_np(x):
    x = x.numpy()
    dummy()
    # x = x * global_y.numpy()
    x = zfit.run(x * global_y)
    # x *= var1.numpy()
    x = np.sqrt(np.abs(x))
    x = np.cos(x - 0.3)
    x = np.sinh(x + 0.4)
    x = x ** 2
    x = np.sum(np.log(x))
    return x


# @tf.function
def calc_tf(x):
    x = tf.sqrt(tf.abs(x))
    x = tf.cos(x - 0.3)
    x = tf.sinh(x + 0.4)
    print(x)
    x = x ** 2
    x = tf.reduce_sum(tf.log(x))
    tf.py_function(dummy, [], Tout=[])
    return x


def calc_torch(x):
    x = torch.tensor(x.numpy())
    x = torch.sqrt(torch.abs(x))
    x = torch.cos(x - 0.3)
    x = torch.sinh(x + 0.4)
    x = x ** 2
    x = torch.sum(torch.log(x))
    return x.numpy()


@tf.function
def calc_np_wrapped(x):
    return tf.py_function(calc_np, [x], Tout=tf.float32)


@tf.function
def calc_torch_wrapped(x):
    return tf.py_function(calc_torch, [x], Tout=tf.float32)


@numba.jit(nopython=True)
def calc_np_numba(x):
    x = np.sqrt(np.abs(x))
    x = np.cos(x - 0.3)
    x = np.sinh(x + 0.4)
    x = x ** 2
    x = np.sum(np.log(x))
    return x


if __name__ == '__main__':
    size = (100000,)
    x = tf.random.normal(shape=size)
    # x = x.numpy()
    # y = zfit.run(calc_tf(x))
    # x = zfit.run(x)
    results = []
    # calc_tf_graph = calc_tf(x)
    # calc_np_wrapped_graph = calc_np_wrapped(x)
    # grad = tf.gradients(calc_np_wrapped_graph, x)
    # grad = tf.gradients(calc_tf_graph, x)
    # print(zfit.run(grad))
    with Timer() as timer:
        for _ in range(100):
            x = tf.random.normal(shape=size)

            # y = calc_np_wrapped(x)
            # y = calc_np(x)
            y = calc_tf(x)
            # y = calc_torch_wrapped(x)
            # y = zfit.run(calc_tf_graph)
            # y = zfit.run(calc_np_wrapped_graph)
            # x = torch.normal(0, 1, size=size)
            # y = calc_torch(x)
            # y = calc_np_numba(x)
            # if not v2behavior:
            #     zfit.run()
            results.append(y)
    print(f"{np.average(results)} +- {np.std(results)}")
    print(timer.elapsed * 1000, 'ms')
