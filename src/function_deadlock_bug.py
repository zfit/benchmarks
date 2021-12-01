import tensorflow as tf


@tf.function(autograph=False)
def func1(depth=0):
    if depth > 1:
        return depth
    else:
        return func1(depth + 1)


func1(0)
