import tensorflow as tf
from zfit_benchmark.timer import Timer

var1 = tf.Variable(42.)


def func(x, y):
    return (x - 1 / y) ** 2 - y ** tf.abs(1 + x)


def func_params(x, y):
    var1.assign(y)
    return func(x=x, y=var1)


def func_args(x, y):
    return func(x, y)


def integrate_func_params(y):
    x = tf.random.uniform(shape=(20000,), minval=-1., maxval=1.)
    return tf.reduce_mean(func_params(x=x, y=y))


def integrate_func_args(y):
    x = tf.random.uniform(shape=(20000,), minval=-1., maxval=1.)
    return tf.reduce_mean(func_args(x=x, y=y))

@tf.function(autograph=False)
def integrate(y, func):
    return tf.map_fn(func, x)
    # return tf.vectorized_map(func, x)


if __name__ == '__main__':
    size = (10000,)
    x = tf.random.normal(shape=size)
    results = []
    n_trials = 10
    with Timer() as timer:
        for _ in range(n_trials):
            x = tf.random.normal(shape=size)
            # integrate(x, integrate_func_args)
            integrate(x, integrate_func_params)

    print(f"Time needed (per run): {timer.elapsed / n_trials :.3} sec")
