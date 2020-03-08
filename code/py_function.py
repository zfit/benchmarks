"""Benchmark of different frameworks for playing around.

no GPU used, serial required (x used in ever next calculation)
| sample_size | tf traced  | tf eager | numpy |
| 1k        | 0.001      | 0.004      | 0.001 |
| 10 k      | 0.005      | 0.013      | 0.007 |
| 100 k     | 0.015      | 0.03       | 0.07  |
| 1 mio     | 0.2        | 0.3        | 0.7   |
| 10 mio    | 2          | 3          | 7     |


no GPU used, parallel possible, list of 10
| sample_size | tf traced  | tf eager | numpy | torch |
| 1k        | 0.0002     | 0.004      | 0.001 | 0.001 |
| 10 k      | 0.0008     | 0.014      | 0.008 | 0.004 |
| 100 k     | 0.002      | 0.04       | 0.08  | 0.02  |
| 1 mio     | 0.02       | 0.4        | 0.8   | 0.3   |
| 10 mio    | 0.2        | 4          | 8     | 3     |
            (with autograph 2 secs)

"""

import numba as numba
import numpy as np
import tensorflow as tf
import torch
from zfit_benchmark.timer import Timer

global_y = tf.random.normal(shape=(10, 1))
var1 = tf.Variable(42.)

list1 = [1, 2, 3, 4]
size = (10000000,)
n_loops = 10


def dummy():
    normal = np.random.normal(size=100000)
    list1.append(np.sum(normal))


def calc_np(x):
    # x = x.numpy()
    # dummy()
    # x = x * global_y.numpy()
    # x = zfit.run(x * global_y)
    # x *= var1.numpy()
    x_init = x
    list1 = []
    for i in range(n_loops):
        x = np.sqrt(np.abs(x_init))
        x = np.cos(x - 0.3)
        x = np.power(x, i + 1)
        x = np.sinh(x + 0.4)
        x = x ** 2
        x += np.random.normal(size=size)
        x /= np.mean(x)
        x = np.abs(x)
        list1.append(x)
    x = np.sum(list1, axis=0)
    x = np.mean(np.log(x))
    return x


@tf.function(autograph=False)
def calc_tf(x):
    x_init = x
    list1 = []
    for i in tf.range(n_loops):
    # for i in range(n_loops):
        x = tf.sqrt(tf.abs(x_init))
        x = tf.cos(x - 0.3)
        x = tf.pow(x, tf.cast(i + 1, tf.float64))
        x = tf.sinh(x + 0.4)
        # print("calc_tf is being traced")
        x = x ** 2
        x += tf.random.normal(shape=size, dtype=tf.float64)
        x /= tf.reduce_mean(x)
        x = tf.abs(x)
        list1.append(x)
    x = tf.reduce_sum(x, axis=0)
    x = tf.reduce_mean(tf.math.log(x))
    # tf.py_function(dummy, [], Tout=[])
    return x


# @torch.jit.script
def calc_torch(x):
    x_init = x
    list1 = []
    for i in range(n_loops):
        x = torch.sqrt(torch.abs(x_init))
        x = torch.cos(x - 0.3)
        x = torch.pow(x, i + 1)
        x = torch.sinh(x + 0.4)
        x = x ** 2
        x += torch.normal(mean=0, std=0, size=size)
        x /= torch.mean(x)
        x = torch.abs(x)
        list1.append(x)
    list1 = torch.stack(list1)
    x = torch.sum(list1, dim=0)
    x = torch.mean(torch.log(x))
    return x.numpy()


@tf.function
def calc_np_wrapped(x):
    return tf.py_function(calc_np, [x], Tout=tf.float32)


@tf.function
def calc_torch_wrapped(x):
    return tf.py_function(calc_torch, [x], Tout=tf.float32)


@numba.jit(nopython=True)
def calc_np_numba(x):
    for i in range(n_loops):
        x = np.sqrt(np.abs(x))
        x = np.cos(x - 0.3)
        x = np.power(x, i)
        x = np.sinh(x + 0.4)
        x = x ** 2
        x = np.mean(np.log(x))
        x += np.random.normal(size=size)
    return x


if __name__ == '__main__':
    x_tf = tf.random.normal(shape=size, dtype=tf.float64)
    x_torch = torch.normal(mean=0, std=0, size=size)
    # x = x.numpy()
    # y = zfit.run(calc_tf(x))
    # x = zfit.run(x)
    results = []
    # calc_tf_graph = calc_tf(x)
    # calc_np_wrapped_graph = calc_np_wrapped(x)
    # grad = tf.gradients(calc_np_wrapped_graph, x)
    # grad = tf.gradients(calc_tf_graph, x)
    # print(zfit.run(grad))
    # x = np.random.normal(size=size)
    y = calc_np(x_tf)
    y = calc_tf(x_tf)
    y = calc_torch(x_torch)

    # y = calc_np_numba(x)

    with Timer() as timer:
        n_runs = 5
        for _ in range(n_runs):
            # x = tf.random.normal(shape=size)
            # with tf.GradientTape() as tape:
            #     tape.watch(x)
            # y = calc_np_wrapped(x)
            # y = calc_np(x_tf)
            # y = calc_tf(x_tf)
            y = calc_torch(x_torch)
            # y = calc_torch_wrapped(x)
            # y = zfit.run(calc_tf_graph)
            # y = zfit.run(calc_np_wrapped_graph)
            # x = torch.normal(0, 1, size=size)
            # y = calc_np_numba(x)
            # if not v2behavior:
            #     zfit.run()
            # gradients = tape.gradient(y, x)
            # print(gradients)
            results.append(y)
    print(f"{np.average(results)} +- {np.std(results)}")
    print(f"Time needed: {timer.elapsed / n_runs :.3} sec")
