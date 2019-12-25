import tensorflow as tf
import numpy as np
from zfit_benchmark.timer import Timer

v2behavior = True
# v2behavior = False
if v2behavior:
    tf.enable_v2_behavior()
import zfit

if not v2behavior:
    tf.disable_eager_execution()

# @tf.function
def func1(x):
    return tf.reduce_sum(tf.sqrt(x))

# @tf.function
def grad(x):
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = func1(x)
    return tape.gradient(y, x)

results = []
if __name__ == '__main__':
    with Timer() as timer:
        for _ in range(100):
            x = tf.random.uniform(shape=(1000000,))
            y = grad(x)
            results.append(y.numpy())
    print(f"{np.average(results)} +- {np.std(results)}")
    print(timer.elapsed * 1000, 'ms')
