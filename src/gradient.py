import tensorflow as tf
import numpy as np
from zfit_benchmark.timer import Timer
import zfit.z.numpy as znp

# @tf.function
def func1(x):
    p = [2.0, 3.0, 4.0, 1.5, 3]
    return tf.reduce_sum(x**5 * p[0] + tf.math.special.dawsn(p[1]*x**2) + p[2]*x**3 + p[3]*x**4 + p[4]*x**5)

# @tf.function
def grad(x):

    with tf.GradientTape() as tape:
        tape.watch(x)
        y = func1(x)
    return tape.gradient(y, x)

# @tf.function
size = (100,)
params = [tf.Variable(val, dtype=znp.float64) for val in znp.random.uniform(low=0., high=10., size=size)]

def func(x):
    return znp.sum([tf.cast(p, tf.float64) ** tf.cast(znp.random.uniform(low=0, high=10, size=[100_000]), znp.float64) for p in x])
def hessian(params):
    with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
        tape.watch(params)
        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
            tape.watch(params)
            y = func(params)
        gradients = tape.gradient(y, params)
    if hessian != 'diag':
        gradients_tf = znp.stack(gradients)
    if hessian == 'diag':
        computed_hessian = znp.stack(
            [tape.gradient(grad, sources=param) for param, grad in zip(params, gradients)]
        )
        # gradfunc = lambda par_grad: tape.gradient(par_grad[0], sources=par_grad[1])
        # computed_hessian = tf.vectorized_map(gradfunc, zip(params, gradients))
    else:
        computed_hessian = znp.asarray(tape.jacobian(gradients_tf, sources=params,
                                                             experimental_use_pfor=False  # causes TF bug? Slow..
                                                             ))
    return computed_hessian

results = []
if __name__ == '__main__':
    with Timer() as timer:
        for _ in range(100):
            # x = tf.random.uniform(shape=(100,))
            y = hessian(params)
            results.append(y.numpy())
    print(f"{np.average(results)} +- {np.std(results)}")
    print(timer.elapsed * 1000, 'ms')
    # gradfunc = lambda par_grad: tape.gradient(par_grad[0], sources=par_grad[1])
    # computed_hessian = tf.vectorized_map(gradfunc, zip(params, gradients))
