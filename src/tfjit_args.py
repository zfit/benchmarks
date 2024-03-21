import tensorflow as tf


@tf.function
def func(x, *args):
    print("Compiling...")
    # tf.print("Compiled, args:", args)
    result = x + tf.reduce_sum(args)
    # tf.print("Result:", result)
    return result


print("Number 1")
func(tf.constant(1), tf.constant(2), tf.constant(3))
print("Number 2")
func(tf.constant(1), tf.constant(2), tf.constant(3), tf.constant(4))
print("Number 3")
func(tf.constant(1), tf.constant(2), tf.constant(3), tf.constant(4), tf.constant(5))
print("Number 4")

func(tf.constant(1), tf.constant(2), tf.constant(3), tf.constant(4))
print("Number 5")
func(tf.constant(1), tf.constant(21), tf.constant(32), tf.constant(5))

