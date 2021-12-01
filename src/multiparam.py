import time

import numpy as np
import progressbar
import tensorflow as tf
import tensorflow.experimental.numpy as tnp

tnp.experimental_enable_numpy_behavior()

nparams = 150
var1 = tf.Variable(np.linspace(0, 10, nparams), dtype=tf.float64, validate_shape=False)
from tensorflow import Variable

class IndexedVariable(Variable):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.active_indices = tuple(range(self.shape[0]))

    def value(self):
        return super.value()[self.active_indices]

    def gather_nd(self, indices, name=None):
        raise RuntimeError
        return super().gather_nd(indices, name)

    def sparse_read(self, indices, name=None):
        raise RuntimeError
        return super().sparse_read(indices, name)


# varindex = IndexedVariable(np.linspace(0, 10, nparams), dtype=tf.float64, validate_shape=False)
indices_all = np.array(range(nparams))
indices = np.random.choice(indices_all, nparams, replace=False)


def model(var):
    var = tnp.array(var)
    # indices = np.array(range(1000))
    var = var[indices]
    return tf.math.special.dawsn(tf.math.abs(var) ** 2.5) * var * tf.math.abs(tnp.cos(var)) ** (var + 0.2) + var ** 3


@tf.function(autograph=False)
def get_hessian2():
    var = vars2
    var = [v.value() for v in var]
    with tf.GradientTape(persistent=True) as tape:
        preds = model(var)
        grads = tape.gradient(preds, var)
        grads = tf.stack(grads, axis=0)
        grads = grads[indices]
    hessians = tape.jacobian(grads, var)
    # hessians = tf.stack(hessians)[:, indices]
    hessians = tf.stack(hessians)
    # hessians = None
    return grads, hessians

@tf.function(autograph=False)
def get_hessian2_classy():
    var = vars2
    with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
        tape.watch(var)
        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape2:
            tape2.watch(var)
            preds = model(var)
        grads = tape2.gradient(preds, var)
        grads = tf.stack(grads, axis=0)
        grads = grads[indices]
    hessians = tape.jacobian(grads, var, experimental_use_pfor=True)
    # hessians = tf.stack(hessians)[:, indices]
    hessians = tf.stack(hessians)
    # hessians = None
    return grads, hessians

@tf.function(autograph=False)
def get_hessian1():
    var = var1
    # var = var[indices]
    # var = [v.value() for v in var]
    with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
        tape.watch(var)
        preds = model(var)
        grads = tape.gradient(preds, var)
        grads = grads[indices]
    hessians = tape.jacobian(grads, var, experimental_use_pfor=True)
    hessians = hessians[:, indices]
    return grads, hessians


with tf.GradientTape(watch_accessed_variables=False) as tape:
    y = var1.sparse_read(5) * 5.
grad = tape.gradient(y, var1.sparse_read(5))
print(grad)

class MyVar(tf.Variable):
    pass
vars2 = [MyVar(val, dtype=tf.float64, validate_shape=False)
         for val in np.linspace(0, 10, nparams)]


def assign2(values, variables):
    for i, var in enumerate(variables):
        var.assign(values[i], use_locking=False, read_value=False)


def assign1(values, variables: tf.Variable):
    variables.assign(values, use_locking=False, read_value=False)
    # updates = tf.IndexedSlices(values[indices], indices)
    # variables.scatter_update(updates, use_locking=False)


assign2_compiled = tf.function(assign2, autograph=False)
assign1_compiled = tf.function(assign1, autograph=False)

start = None
prev = 1
for nrun in progressbar.progressbar(range(100)):
    if nrun > 2 and start is None:
        start = time.time()
        print('starting the time')
    uniform = tnp.random.uniform(size=(nparams,))
    result = get_hessian2()
    # result = get_hessian2_classy()
    assign2_compiled(uniform, vars2)
    # result = get_hessian1()
    # assign1_compiled(values=uniform, variables=var1)
    # assign1(values=uniform, variables=var1)
    # assign2(uniform, vars2)
    # assign(uniform, vars2)

print(f'time per param needed: {(time.time() - start) / nparams}')
print(result)
