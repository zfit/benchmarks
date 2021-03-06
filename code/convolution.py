import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import tensorflow as tf

# n = 101
n = 100
array1 = np.random.uniform(0, 5, size=(n, n))
array1b = np.random.uniform(0, 5, size=(n, n))
array2 = np.random.normal(loc=1, scale=2, size=(n, n))
array2b = np.random.normal(loc=1, scale=2, size=(n, n))

lower1 = 0
upper1 = 10
linspace1 = np.linspace(lower1, upper1, num=n // 2 + 1)
x1 = np.concatenate([linspace1, np.zeros(shape=n // 2)])
upper2 = 40
lower2 = 20
linspace2 = np.linspace(lower2, upper2, num=n // 2 + 1)
x2 = np.concatenate([linspace2, np.ones(shape=n // 2)])

linspace1 = np.linspace(lower1, upper1, num=n // 2 + 1)
xk1 = np.concatenate([linspace1[::-1], np.zeros(shape=n // 2)])
linspace2 = np.linspace(lower2, upper2, num=n // 2 + 1)
xk2 = np.concatenate([3 * np.ones(shape=n // 4), linspace2, np.ones(shape=n // 4)])

linfull1 = np.linspace(lower1, upper1, num=n + 1)
linfull2 = np.linspace(lower2, upper2, num=n + 1)
ar = tf.transpose(tf.meshgrid(linfull1, linfull2, indexing='ij'))
# ark1, ark2 =
# ark1, ark2 = np.meshgrid(xk2, xk1, indexing='ij')

# array1 = x1 * x2[..., None]
# xk = xk1 * xk2[..., None]
# array2 = xk
n = n + 1
array1 = (lambda x, y: tf.math.square(x) * tf.math.cos(y))(*tf.unstack(ar, axis=-1))
array2 = (lambda x, y: 1. * tf.math.sqrt(y))(*tf.unstack(xk1, xk2, axis=-1))
array1 = tf.reshape(array1, (n, n))
array2 = tf.reshape(array2, (n, n))
# mode = 'same'
mode = 'same'
corr_np = scipy.signal.correlate(array1, array2[::-1, ::-1], mode=mode)
corr2d_np = scipy.signal.correlate2d(array1, array2[::-1, ::-1], mode=mode)
conv_np = scipy.signal.convolve(array1, array2, mode=mode)
# conv_np = scipy.signal.fftconvolve(array1, array2[::-1, ::-1], mode='same')
# conv_np = scipy.signal.convolve2d(array1, -array2, mode='same')
# conv_npb = scipy.signal.correlate(array1b, array2b, mode=mode)
array_rev = tf.reverse(array2, axis=(0, 1))
# array_rev = array2[::-1, ::-1]
conv_tf = tf.nn.convolution(array1[None, ..., None], array_rev[..., None, None,],
                            strides=1, padding='SAME')[0, ..., 0]

diff = corr_np - conv_tf
# diffb = corr_np - conv_npb

plt.figure()
# plt.hist2d(array1, array2)

plt.figure()
plt.title("tf vs np")
plt.imshow(diff)
plt.colorbar()

plt.figure()
plt.title("convolution")
plt.imshow(conv_tf)
plt.colorbar()

# plt.figure()
# plt.title("corr vs conv")
# plt.imshow(corr_np - conv_np)
# plt.colorbar()

plt.figure()
plt.title("corr vs corr2d")
plt.imshow(corr_np - corr2d_np)
plt.colorbar()

plt.figure()
plt.title("tf vs corr2d")
plt.imshow(conv_tf - corr2d_np)
plt.colorbar()

plt.show()
