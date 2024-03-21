"""Test the performance of different cache methods


Comparison of a feed_dict based approach and an approach based on a Variable actings as cache.

Results:
(100 runs)
non-cached:
variable: 7.3 sec
feed_dict: 7.4 sec

cached:
variable: 0.02 sec
feed_dict: 0.026 sec

"""

import numpy as np
import tensorflow as tf
from zfit import z
from zfit_benchmark.timer import Timer

# zfit.run.set_graph_mode(False)

do_cache = True
# do_cache = False
rnd_prob = 0.0  # how often to randomly invalidate the cache
# sanity check: if ~1, should behave like no cache, if ~0, nearly no std and fast
# setting it to zero means no invalidation ever


z.function


def func_a(x):
    return tf.math.log(tf.math.exp(x - 0.01) * 1.01 + 0.1) * 0.99 - tf.math.sin(x * 0.98)


z.function


def func_b(x):
    return tf.math.cos(tf.math.exp(x - 0.011) * 1.03 + 0.11) * 0.992 - tf.math.sin(x * 0.984)


def expensive(func):
    return tf.math.reduce_mean(func(tf.random.uniform(shape=(10000000,))))


class BaseModel():
    def __init__(self) -> None:
        self.cache = {}
        super().__init__()

    def value(self):
        return self.expensive_a() + self.expensive_b()

    def expensive_a(self):
        raise NotImplementedError

    def expensive_b(self):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError


# OLD TensorFlow 1
# class FeedModel(BaseModel):
# 
#     def expensive_a(self):
#         val = expensive(func_a)
#         return val
# 
#     def expensive_b(self):
#         val = expensive(func_b)
#         return val
# 
#     def run(self):
#         val, a, b = self.sess.run([self.value, self.a, self.b], feed_dict=self.cache)
#         if do_cache:
#             if not (rnd_cache and np.random.choice([True, False], p=[0.5, 0.5])):
#                 self.cache[self.a] = a
#             else:
#                 with suppress(KeyError):
#                     del self.cache[self.a]
#             if not (rnd_cache and np.random.choice([True, False], p=[0.5, 0.5])):
#                 self.cache[self.b] = b
#             else:
#                 with suppress(KeyError):
#                     del self.cache[self.b]
#         return val


def expensive_auto_cache(cache: tf.Variable, flag: tf.Variable, func):
    def autoset_func():
        val = func()

        cache.assign(val, read_value=False)
        flag.assign(True, read_value=False)
        return cache

    val = tf.cond(flag, lambda: cache, autoset_func)
    return val


class VariableModel(BaseModel):

    def __init__(self):
        super().__init__()
        self.is_cached = {}
        self.cache['a'] = tf.Variable(initial_value=42., trainable=False)
        self.is_cached['a'] = tf.Variable(initial_value=False, trainable=False)
        self.cache['b'] = tf.Variable(initial_value=42., trainable=False)
        self.is_cached['b'] = tf.Variable(initial_value=False, trainable=False)

    @z.function()
    def expensive_a(self):
        return expensive_auto_cache(cache=self.cache['a'], flag=self.is_cached['a'], func=lambda: expensive(func_a))

    @z.function
    def expensive_b(self):
        return expensive_auto_cache(cache=self.cache['b'], flag=self.is_cached['b'], func=lambda: expensive(func_b))

    def run(self):
        if not do_cache or (do_cache and np.random.choice([True, False],
                                                          p=[rnd_prob, 1 - rnd_prob])):
            self.is_cached['a'].assign(False)
        if not do_cache or (do_cache and np.random.choice([True, False],
                                                          p=[rnd_prob, 1 - rnd_prob])):
            self.is_cached['b'].assign(False)
        return self.value()


if __name__ == '__main__':
    n_runs = 100
    values = np.zeros(shape=(n_runs,))

    model = VariableModel()
    model.run()  # pre run to remove possible initial overhead, caches also values
    with Timer() as timer:
        for i in range(n_runs):
            values[i] = model.run()
    print(f"mean={np.mean(values):.4g} +- {np.std(values):.4g}")
    print(f"{timer.elapsed:.3f} sec")
