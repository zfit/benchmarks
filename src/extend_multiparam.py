import tensorflow as tf
import progressbar
import time

nparams = 1000
start = None
prev = 1
values = tf.linspace(0, 10, nparams)
var1 = tf.Variable([-1], dtype=tf.float64, shape=tf.TensorShape(None), validate_shape=False)

for nrun in progressbar.progressbar(range(10)):
    if nrun > 2 and start is None:
        start = time.time()
    for newval in values:
        val = var1.value()
        newvar = tf.concat([val, [newval]], axis=0)
        var1.assign(newvar, use_locking=False, read_value=False)

print(f'time per param needed: {(time.time() - start) / nparams}')
print(var1.value())
