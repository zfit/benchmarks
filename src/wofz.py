# /////////////////////////////////////////////////////////////////////////////
# //
# // DATE
# //   06/22/2015
# //
# // AUTHORS
# //   Hannes Bartosik, Adrian Oeftiger
# //
# // DESCRIPTION
# //   FADDEEVA error function for GPU in CUDA.
# //   This file is intended to be used as a
# //   preamble to depending kernels, e.g. in PyCUDA
# //   via ElementwiseKernel(..., preamble=open( <this_file> ).read()).
# //
# /////////////////////////////////////////////////////////////////////////////

# include <math.h>
import time

errf_const = 1.12837916709551
xLim = 5.33
yLim = 4.29

import tensorflow.experimental.numpy as znp

znp.experimental_enable_numpy_behavior()

# from tensorflow.experimental.numpy import *
import tensorflow as tf
from math import sqrt, exp, cos, sin


@tf.function
def wofz2(in_real, in_imag):
    # /**
    # this function calculates the double precision complex error function
    # based on the algorithm of the FORTRAN function written at CERN by
    # K. Koelbig, Program C335, 1970.
    #
    # See also M. Bassetti and G.A. Erskine, "Closed expression for the
    # electric field of a two-dimensional Gaussian charge density",
    # CERN-ISR-TH/80-06.
    # */

    x = abs(in_real)
    y = abs(in_imag)

    cond = znp.logical_and(y < yLim, x < xLim)
    nevents = tf.shape(x)[0]

    def if_true():
        # Rx = znp.zeros([nevents, 33], dtype=znp.float64)
        # Ry = znp.zeros([nevents, 33], dtype=znp.float64)
        q = (1.0 - y / yLim) * sqrt(1.0 - (x / xLim) * (x / xLim))
        h = 1.0 / (3.2 * q)
        nc = 7 + tf.cast(23.0 * q, dtype=znp.int32)
        xl = pow(h, 1. - nc)
        xh = y + 0.5 / h
        yh = x
        nu = 10 + tf.cast(21.0 * q, dtype=znp.int32)
        Rx = znp.zeros_like(x, dtype=znp.float64)
        Ry = znp.zeros_like(y, dtype=znp.float64)
        n = nu

        n2 = nc

        # rxs = []
        # rys = []

        Sx = znp.zeros_like(x, dtype=znp.float64)
        Sy = znp.zeros_like(x, dtype=znp.float64)
        while znp.any(n > 0):
            n = znp.maximum(n, 0)
            Tx = xh + n * Rx
            Ty = yh - n * Ry
            Tn = Tx * Tx + Ty * Ty
            # indices = znp.asarray([tf.range(nevents), n - 1])
            # Rx = tf.transpose(Rx)
            # Ry = tf.transpose(Ry)
            # Rx = tf.tensor_scatter_nd_update(Rx, [n - 1], (0.5 * Tx / Tn))
            # Ry = tf.tensor_scatter_nd_update(Ry, [n - 1], (0.5 * Ty / Tn))
            # Rx = tf.transpose(Rx)
            # Ry = tf.transpose(Ry)
            Rx = (0.5 * Tx / Tn)
            Ry = (0.5 * Ty / Tn)

            Saux = Sx + xl
            indices = znp.stack([n - 1, tf.range(n.shape[0])], axis=1)
            mask = tf.cast(n2 == n, dtype=float64)
            rx_n1 = Rx * mask
            ry_n1 = Ry * mask
            Sx_tmp = rx_n1 * Saux - ry_n1 * Sy
            Sy_tmp = rx_n1 * Sy + ry_n1 * Saux
            cond_inside = n > 0
            Sx = znp.where(cond_inside, Sx_tmp, Sx)
            Sy = znp.where(cond_inside, Sy_tmp, Sy)
            xl = h * xl
            n -= 1
            n2 = tf.maximum(n, n2 - 1)
            print(znp.max(n))

        # Rx = znp.stack(rxs)
        # Ry = znp.stack(rys)
        # # Rx = tf.transpose(Rx)
        # # Ry = tf.transpose(Ry)
        #
        #
        # n = nc
        #
        # while znp.any(n > 0):
        #     n = znp.maximum(n, 0)
        #     Saux = Sx + xl
        #     indices = znp.stack([n - 1, tf.range(n.shape[0])], axis=1)
        #     rx_n1 = tf.gather_nd(Rx, indices)
        #     ry_n1 = tf.gather_nd(Ry, indices)
        #     Sx = rx_n1 * Saux - ry_n1 * Sy
        #     Sy = rx_n1 * Sy + ry_n1 * Saux
        #     xl = h * xl
        #     n -= 1

        Wx = errf_const * Sx
        Wy = errf_const * Sy
        return Wx, Wy

    def if_false():

        xh = y
        yh = x
        rx = znp.zeros_like(x, dtype=znp.float64)
        ry = znp.zeros_like(y, dtype=znp.float64)
        for n in tf.range(1, 10):
            Tx = xh + n * rx
            Ty = yh - n * ry
            Tn = Tx ** 2 + Ty ** 2
            rx = 0.5 * Tx / Tn
            ry = 0.5 * Ty / Tn

        Wx = errf_const * rx
        Wy = errf_const * ry
        return Wx, Wy

    # if y == 0.:
    #     Wx = exp(-x * x)

    cond2 = in_imag < 0.

    def if_true2(Wx, Wy):
        Wx = 2.0 * exp(y * y - x * x) * cos(2.0 * x * y) - Wx
        Wy = - 2.0 * exp(y * y - x * x) * sin(2.0 * x * y) - Wy
        Wy = -Wy * znp.sign(in_real)
        return Wx, Wy

    def if_false2(Wx, Wy):
        return Wx, Wy * znp.sign(in_real)

    value = znp.where(cond, if_true(), if_false())
    true2 = if_true2(*tf.unstack(value))
    false2 = if_false2(*tf.unstack(value))
    value = znp.where(cond2, true2, false2)
    return value[0] + 1j * value[1]


errf_const = 1.12837916709551
xLim = 5.33
yLim = 4.29
#
# __device__ void wofz(double in_real, double in_imag,
#                      double* out_real, double* out_imag)

# /**
# this function calculates the double precision complex error function
# based on the algorithm of the FORTRAN function written at CERN by
# K. Koelbig, Program C335, 1970.
# See also M. Bassetti and G.A. Erskine, "Closed expression for the
# electric field of a two-dimensional Gaussian charge density",
# CERN-ISR-TH/80-06.
# */

# int n, nc, nu
# double h, q, Saux, Sx, Sy, Tn, Tx, Ty, Wx, Wy, xh, xl, x, yh, y
import numba


@numba.vectorize()
def wofz(in_real, in_imag) -> complex:
    Rx = []
    Ry = []

    x = abs(in_real)
    y = abs(in_imag)

    if (y < yLim and x < xLim):
        q = (1.0 - y / yLim) * sqrt(1.0 - (x / xLim) * (x / xLim))
        h = 1.0 / (3.2 * q)
        nc = 7 + int(23.0 * q)
        xl = pow(h, 1. - nc)
        xh = y + 0.5 / h
        yh = x
        nu = 10 + int(21.0 * q)
        Rx[nu] = 0.
        Ry[nu] = 0.
        n = nu
        while (n > 0):
            Tx = xh + n * Rx[n]
            Ty = yh - n * Ry[n]
            Tn = Tx * Tx + Ty * Ty
            Rx[n - 1] = 0.5 * Tx / Tn
            Ry[n - 1] = 0.5 * Ty / Tn
            n -= 1

        Sx = 0.
        Sy = 0.
        n = nc
        while n > 0:
            Saux = Sx + xl
            Sx = Rx[n - 1] * Saux - Ry[n - 1] * Sy
            Sy = Rx[n - 1] * Sy + Ry[n - 1] * Saux
            xl = h * xl
            n -= 1

        Wx = errf_const * Sx
        Wy = errf_const * Sy

    else:
        xh = y
        yh = x
        Rx[0] = 0.
        Ry[0] = 0.
        for n in tf.range(9, 0, -1):
            Tx = xh + n * Rx[0]
            Ty = yh - n * Ry[0]
            Tn = Tx * Tx + Ty * Ty
            Rx[0] = 0.5 * Tx / Tn
            Ry[0] = 0.5 * Ty / Tn

        Wx = errf_const * Rx[0]
        Wy = errf_const * Ry[0]

    if (y == 0.):
        Wx = exp(-x * x)

    if (in_imag < 0.):
        Wx = 2.0 * exp(y * y - x * x) * cos(2.0 * x * y) - Wx
        Wy = - 2.0 * exp(y * y - x * x) * sin(2.0 * x * y) - Wy
        if (in_real > 0.):
            Wy = -Wy
    elif (in_real < 0.):
        Wy = -Wy


if __name__ == '__main__':
    import scipy.special
    import numpy as np

    wofz(
        # znp.array([10.], dtype=znp.float64), znp.array([5.], dtype=znp.float64))
        *np.random.uniform(-10, 10, (2, 1000000)))
    print("compiled")
    start = time.time()
    x = np.random.uniform(-10, 10, (2, 1000000))
    n = 10
    for _ in range(n):
        wofz_our = wofz(
            # znp.array([10.], dtype=znp.float64), znp.array([5.], dtype=znp.float64))
            *x
        )
    print('tensorflow', time.time() - start)
    x = x[0] + 1j * x[1]
    start = time.time()
    for _ in range(n):
        y = scipy.special.wofz(x)
    print('scipy', time.time() - start)

    print(abs(wofz_our - y), znp.std(wofz_our - y))
