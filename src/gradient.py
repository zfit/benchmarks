import progressbar
import tensorflow as tf
import numpy as np
from zfit import z

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
size = (10,)
params = [tf.Variable(val, dtype=znp.float64) for val in znp.random.uniform(low=4., high=14., size=size)]

@tf.function
def func(x):
    return znp.sum([tf.cast(p, tf.float64) ** (tf.cast(p2, tf.float64) * tf.cast(znp.random.uniform(low=3, high=10, size=[100_000]), znp.float64))
                    for p2 in tf.unstack(x) for p in tf.unstack(x)])

# @tf.function
def hessian(params, hess):

    params = tf.stack(params)
    with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
        tape.watch(params)
        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape2:
            tape2.watch(params)
            y = func(params)
        gradients = tape2.gradient(y, params)
        gradients_params = tf.stack([gradients, params])

    # if hess != 'diag':
    #     gradients_tf = znp.stack(gradients)
    if hess == 'diag':
        # computed_hessian = znp.stack([tape.gradient(grad, sources=param) for param, grad in zip(params, gradients)])
        def gradfunc(par_grad):
            pars = par_grad[1]
            grads = par_grad[0]
            tf.print(pars)
            tf.print(grads)
            return tape.gradient(grads, sources=pars)
        # gradfunc = lambda par_grad: par_grad[1]
        computed_hessian = tf.map_fn(gradfunc, gradients_params)
        # computed_hessian = tf.map_fn(gradfunc, list(zip(params, gradients)))
    else:
        computed_hessian = znp.asarray(tape.jacobian(gradients_tf, sources=params,
                                                     experimental_use_pfor=True  # causes TF bug? Slow..
                                                     ))
    return computed_hessian

def hessian(x, hess):
    x = tf.stack(x)
    with tf.GradientTape(persistent=True) as tape:
        y = func(x)
        grads = tape.gradient(y, x)
        flattened_grads = tf.concat([tf.reshape(grad, [-1]) for grad in grads], axis=0)
    hessians = tape.jacobian(flattened_grads, x)
    flattened_hessians = tf.concat([tf.reshape(hess, [hess.shape[0], -1]) for hess in hessians], 1)
    return flattened_hessians

results = []
if __name__ == '__main__':
    hess = 'diag'
    y = hessian(params, hess=hess)
    # y = hessian(params)
    with Timer() as timer:
        nruns = 10
        for _ in progressbar.progressbar(range(nruns)):
            # x = tf.random.uniform(shape=(100,))
            y = hessian(params, hess=hess)
            results.append(y.numpy())
    print(f"{np.average(results, axis=0)} +- {np.std(results, axis=0)}")
    print(timer.elapsed * 1000 / nruns, 'ms')
    # gradfunc = lambda par_grad: tape.gradient(par_grad[0], sources=par_grad[1])
    # computed_hessian = tf.vectorized_map(gradfunc, zip(params, gradients))
