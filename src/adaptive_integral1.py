import math

import tensorflow as tf
from zfit import z


def integrate(func, lower, upper):
    func_upper = func(upper)
    func_lower = func(lower)
    uncertainty = tf.abs(func_upper - func_lower)  # can be improved of course
    integrals = (func_lower + func_upper) / 2 * (upper - lower)
    return integrals, uncertainty


# func = lambda x: tf.where(tf.less(x, 0.1),
#                           tf.sin(x * 100),
#                           tf.sin(x))
func = lambda x: tf.sin(x) + tf.cos(x * 100)  # example func to integrate
lower, upper = z.constant(0.), z.constant(math.pi)

n_iter_max = 32  # maximum iteration: if we have a discontinuous function, we won't reacht the precision requested


# so we should break


def body(integral, lower, upper, n_iter):
    integrals, uncertainties = integrate(func, lower, upper)
    uncertainties_too_large = tf.greater(uncertainties, 1e-5)
    # if we reached the max number of iterations, we take the values anyway, so the uncertainties are just "too large",
    # or need to be redone if we did not yet reach the max iterations
    uncertainties_too_large = tf.logical_and(uncertainties_too_large, n_iter < n_iter_max)

    too_large_indices = tf.where(uncertainties_too_large)[:, 0]
    # tf.print(integrals[:5])
    # tf.print(uncertainties[:5])
    # tf.print(too_large_indices[:5])

    integral += tf.reduce_sum(tf.boolean_mask(integrals, mask=tf.logical_not(uncertainties_too_large)), axis=0)
    tf.print(integral)

    lower_to_redo = tf.gather(lower, too_large_indices, axis=0)  # take the indices of the lower that need to be redone
    # tf.print(lower_to_redo[:5])
    upper_to_redo = tf.gather(upper, too_large_indices, axis=0)
    # tf.print(upper_to_redo[:5])
    new_middle = (upper_to_redo + lower_to_redo) / 2  # create points in the middle of the current lower, upper
    # the new points are now: old lower, and new middle points respectively new middle point and old upper
    new_lower = tf.concat([lower_to_redo, new_middle], axis=0)
    # tf.print(new_lower[:5])
    new_upper = tf.concat([new_middle, upper_to_redo], axis=0)
    # tf.print(new_upper[:5])
    return integral, new_lower, new_upper, n_iter + 1


def all_calculated(integral, lower, upper, n_iter):
    shape = tf.shape(lower)[0]  # number of integrals to redo. If this is 0, we're fine
    tf.print(shape)
    return tf.logical_and(shape > 0, n_iter < n_iter_max)


initial_points = tf.linspace(lower, upper, num=101)  # start with som initial points


@tf.function(autograph=False)
def do_integrate():
    return tf.while_loop(cond=all_calculated, body=body, loop_vars=[z.constant(0.),  # integral
                                                                    initial_points[:-1],  # lower
                                                                    initial_points[1:],  # upper
                                                                    0  # n_iter
                                                                    ],
                         # here we specify the shape of the loop_vars: since they change (of the second and third),
                         # we need to specify them, with None as "shape is not fixed". For the integral as well as for
                         # the number of iterations, this is a scalar with shape ()
                         shape_invariants=[
                             tf.TensorShape(()),
                             tf.TensorShape((None,)),
                             tf.TensorShape((None,)),
                             tf.TensorShape(()),
                         ],
                         maximum_iterations=n_iter_max,
                         )


integral = do_integrate()
print(integral[0])
