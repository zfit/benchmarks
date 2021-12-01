import progressbar
import tensorflow as tf
from zfit_benchmark.timer import Timer

var1 = tf.Variable(42., dtype=tf.float32, validate_shape=False, shape=tf.TensorShape(None))

size_int_sample = 20000


@tf.function(autograph=False)
def func(x, y):
    return (x - 1 / (y + 100)) ** 2 - y * tf.abs(1 + x) * tf.cos(x) * tf.sin(y ** 2) * tf.math.erfc(tf.abs(x + 0.1)) * tf.math.special.dawsn(tf.cos(x) ** 2)


def func_params(x, y):
    var1.assign(y)
    # tf.assign(var1, y)
    vals = func(x=x, y=var1)
    tf.debugging.assert_equal(var1.value(), y)
    # print(vals)
    return vals


def func_args(x, y):
    vals = func(x, y)
    tf.debugging.assert_equal(y + 1, y + 1)
    # print(vals)
    return vals


def integrate_func_params(y):
    x = tf.random.uniform(shape=(size_int_sample,), minval=-1., maxval=1.)
    return tf.reduce_mean(func_params(x=x, y=y))


def integrate_func_args(y):
    x = tf.random.uniform(shape=(size_int_sample,), minval=-1., maxval=1.)
    return tf.reduce_mean(func_args(x=x, y=y), axis=-1)


@tf.function(autograph=False)
def integrate(y, func):
    # vals = tf.map_fn(func, y, parallel_iterations=14)
    vals = tf.vectorized_map(func, y)
    print(vals)
    return vals


@tf.function(autograph=False)
def integrate_broadcast(y, func):
    y = y[:, None]
    func1 = func(y)
    print(func1)
    return func1
    # return tf.vectorized_map(func, x)


if __name__ == '__main__':
    # tf.config.experimental_run_functions_eagerly(True)
    size = (10000,)
    x = tf.random.normal(mean=10., shape=size)
    results = []
    n_trials = 2

    # import multiprocessing as mp
    # pool = mp.pool.Pool(2)
    # xs = [tf.random.normal(mean=10., shape=size) for _ in range(2)]
    # def pfunc(x):
    #     return integrate(x, integrate_func_args)
    # results = pool.map(func, xs)
    #
    logdir = 'tmp_logresults'


    # writer = tf.summary.create_file_writer(logdir)
    # tf.summary.trace_on(graph=True, profiler=True)

    @tf.function
    def func1(x, y):
        return x + y


    with tf.profiler.experimental.Profile(logdir):
        # Train the model here

        func1(tf.constant(1), tf.constant(41))
    with Timer() as timer:
        timer.stop()
        x = tf.random.normal(shape=size)

        for i in progressbar.progressbar(range(n_trials + 1)):
            if i == 1:
                timer.start()
            #         with tf.device('/device:cpu:0'):
            # result = pfunc(x)
            # result = integrate(x, integrate_func_args)
            result = integrate_broadcast(x, integrate_func_args)
            # result = integrate(x, integrate_func_params)
            # result = integrate_broadcast(x, integrate_func_params)
    #
    if n_trials > 0:
        print(f"Result = {result}")
        print(f"Time needed (per run): {timer.elapsed / n_trials :.3} sec")

    # with writer.as_default():
    #     tf.summary.trace_export('params_vs_argument', step=0, profiler_outdir=logdir)
