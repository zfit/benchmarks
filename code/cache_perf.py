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
from contextlib import suppress

import tensorflow as tf
import numpy as np

from zfit_benchmark.timer import Timer

# variable_cache = True
variable_cache = False
do_cache = True
# do_cache = False
rnd_cache = False
# rnd_cache = True
# if not variable_cache:
tf.compat.v1.disable_eager_execution()

def func_a(x):
    return tf.log(tf.exp(x - 0.01)* 1.01 + 0.1) * 0.99 - tf.sin(x*0.98)


def func_b(x):
    return tf.cos(tf.exp(x - 0.011)* 1.03 + 0.11) * 0.992 - tf.sin(x*0.984)


def expensive(func):
    return tf.reduce_mean(func(tf.random.uniform(shape=(10000000,))))


class BaseModel():
    def __init__(self, sess) -> None:
        self.sess = sess
        self.cache = {}
        self.a = self.expensive_a()
        self.b = self.expensive_b()
        self.value = self.a + self.b
        super().__init__()

    def expensive_a(self):
        raise NotImplementedError

    def expensive_b(self):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError


class FeedModel(BaseModel):

    def expensive_a(self):
        val = expensive(func_a)
        return val

    def expensive_b(self):
        val = expensive(func_b)
        return val

    def run(self):
        val, a, b = self.sess.run([self.value, self.a, self.b], feed_dict=self.cache)
        if do_cache:
            if not (rnd_cache and np.random.choice([True, False], p=[0.5, 0.5])):
                self.cache[self.a] = a
            else:
                with suppress(KeyError):
                    del self.cache[self.a]
            if not (rnd_cache and np.random.choice([True, False], p=[0.5, 0.5])):
                self.cache[self.b] = b
            else:
                with suppress(KeyError):
                    del self.cache[self.b]
        return val


def expensive_auto_cache(cache, flag, func):
    def autoset_func():
        val = func()
        assign_cache_op = cache.assign(val)
        assign_flag_op = flag.assign(True)
        with tf.control_dependencies([assign_cache_op]):
            # avoid race conditions
            with tf.control_dependencies([assign_flag_op]):
                return tf.identity(val)

    val = tf.cond(flag, lambda: cache, autoset_func)
    return val


class VariableModel(BaseModel):

    def __init__(self, sess):
        self.is_cached = {}
        super().__init__(sess)

    def expensive_a(self):
        self.cache['a'] = tf.Variable(initial_value=42., trainable=False, use_resource=True)
        self.is_cached['a'] = tf.Variable(initial_value=False, trainable=False, use_resource=True)
        self.sess.run(self.cache['a'].initializer)
        self.sess.run(self.is_cached['a'].initializer)

        return expensive_auto_cache(cache=self.cache['a'], flag=self.is_cached['a'], func=lambda: expensive(func_a))

    def expensive_b(self):
        self.cache['b'] = tf.Variable(initial_value=42., trainable=False, use_resource=True)
        self.is_cached['b'] = tf.Variable(initial_value=False, trainable=False, use_resource=True)
        self.sess.run(self.cache['b'].initializer)
        self.sess.run(self.is_cached['b'].initializer)
        return expensive_auto_cache(cache=self.cache['b'], flag=self.is_cached['b'], func=lambda: expensive(func_b))

    def run(self):
        if not do_cache or (do_cache and rnd_cache and np.random.choice([True, False], p=[0.5, 0.5])):
            self.is_cached['a'].load(False, session=self.sess)
        if not do_cache or (do_cache and rnd_cache and np.random.choice([True, False], p=[0.5, 0.5])):
            self.is_cached['b'].load(False, session=self.sess)
        return self.sess.run(self.value)


if __name__ == '__main__':
    n_runs = 100
    values = []

    with tf.compat.v1.Session() as sess:
        if variable_cache:
            model = VariableModel(sess=sess)
        else:
            model = FeedModel(sess=sess)
        model.run()  # pre run to remove overhead
        with Timer() as timer:
            for _ in range(n_runs):
                values.append(model.run())
        print(np.mean(values), np.std(values))
        print(timer.elapsed)
