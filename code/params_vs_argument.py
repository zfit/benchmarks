import progressbar
import tensorflow as tf
from zfit_benchmark.timer import Timer

var1 = tf.Variable(42.)

size_int_sample = 20000


@tf.function(autograph=False)
def func(x, y):
    return (x - 1 / y) ** 2 - y ** tf.abs(1 + x)


def func_params(x, y):
    var1.assign(y)
    vals = func(x=x, y=var1)
    tf.debugging.assert_equal(var1.value(), y)
    return vals


def func_args(x, y):
    vals = func(x, y)
    tf.debugging.assert_equal(y + 1, y + 1)
    return vals


def integrate_func_params(y):
    x = tf.random.uniform(shape=(size_int_sample,), minval=-1., maxval=1.)
    return tf.reduce_mean(func_params(x=x, y=y))


def integrate_func_args(y):
    x = tf.random.uniform(shape=(size_int_sample,), minval=-1., maxval=1.)
    return tf.reduce_mean(func_args(x=x, y=y), axis=-1)


@tf.function(autograph=False)
def integrate(y, func):
    return tf.map_fn(func, y, parallel_iterations=14)

@tf.function(autograph=False)
def integrate_broadcast(y, func):
    y = y[:, None]
    return func(y)
    # return tf.vectorized_map(func, x)


if __name__ == '__main__':
    size = (50000,)
    x = tf.random.normal(shape=size)
    results = []
    n_trials = 3
    with Timer() as timer:
        for _ in progressbar.progressbar(range(n_trials)):
            x = tf.random.normal(shape=size)
            result = integrate(x, integrate_func_args)
            # result = integrate_broadcast(x, integrate_func_args)
            # result = integrate(x, integrate_func_params)

    print(f"Result = {result}")
    print(f"Time needed (per run): {timer.elapsed / n_trials :.3} sec")
