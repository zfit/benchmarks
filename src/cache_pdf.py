import zfit
import tensorflow as tf
import zfit.z.numpy as znp
from zfit import z


def cache_value(cache: tf.Variable, flag: tf.Variable, func):
    @z.function
    def autoset_func():
        val = func()

        cache.assign(val, read_value=False)
        flag.assign(True, read_value=False)
        tf.print('Function evaluated')
        return cache

    val = tf.cond(flag, lambda: cache, autoset_func)
    return val


class CachedPDF(zfit.pdf.BaseFunctor):
    def __init__(self, pdf, cache_tol=None, do_caching=None, **kwargs):
        if do_caching is None:
            do_caching = True
        super().__init__(pdfs=pdf, obs=pdf.space, **kwargs)
        params = list(pdf.get_params())
        self.param_values = tf.Variable(tf.stack(params) * 0.1, dtype=tf.float64, trainable=False)
        self.valcache = None
        self.valcache_valid = tf.Variable(initial_value=False, trainable=False)
        self.do_caching = tf.Variable(initial_value=False, trainable=False)
        self.cache_tolerance = 1e-8 if cache_tol is None else cache_tol
        self.do_value_caching(do_caching)

    def do_value_caching(self, flag):
        self.do_caching.assign(flag)

    @zfit.supports(norm=True)
    def _pdf(self, x, norm_range):
        valcache = self.valcache
        if valcache is None:
            valcache = tf.Variable(znp.zeros(shape=tf.shape(x)[0]), trainable=False, validate_shape=False,
                                   dtype=tf.float64)
            self.valcache = valcache

        params = list(self.pdfs[0].get_params())
        values = tf.stack(params)
        params_same = tf.math.reduce_all(tf.math.abs(values - self.param_values) < self.cache_tolerance)
        self.valcache_valid.assign(tf.math.logical_and(params_same, self.do_caching), read_value=False)
        self.param_values.assign(values, read_value=False)
        value = cache_value(valcache, self.valcache_valid, lambda: self.pdfs[0].pdf(x, norm_range))
        return znp.asarray(value)


if __name__ == '__main__':
    obs1 = zfit.Space('x', limits=(-5, 5))
    mu = zfit.Parameter('mu', 1.0)
    sigma = zfit.Parameter('sigma', 1.0)
    pdf1 = zfit.pdf.Gauss(mu, sigma, obs=obs1)
    gauss_cached = CachedPDF(pdf1, do_caching=True)

    x = tf.random.uniform(shape=(100000,), minval=-5, maxval=5)
    print("Evaluate 1")
    y = gauss_cached.pdf(x)
    print("Evaluate 2")
    y = gauss_cached.pdf(x)
    print('change params')
    mu.set_value(2.0)
    print("Evaluate 3")
    y = gauss_cached.pdf(x)
    print("Evaluate 4")
    y = gauss_cached.pdf(x)
    print("finished")
