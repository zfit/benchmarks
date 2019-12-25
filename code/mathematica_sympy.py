import sympy
import tensorflow as tf

import sympy.parsing.mathematica as symath


math_expr = '(t^3+10t^2*a*Sin[x]+b*32t+32)/(t^2+2t-15)'
parsed_expr = symath.mathematica(s=math_expr)
print("parsed sympy expression", parsed_expr)
tf_expr = sympy.lambdify(parsed_expr.free_symbols, parsed_expr, 'tensorflow')
data = tf.linspace(0., 10., num=10)
print(tf_expr)
a = tf.Variable(15.)
b = tf.Variable(13.)
with tf.Session() as sess:
    sess.run([a.initializer, b.initializer])
    tensor = tf_expr(a=a, b=b, x=data, t=data)
    print(sess.run(tensor))
    a.load(150., session=sess)
    print(sess.run(tensor))
    a.load(15., session=sess)
    print(sess.run(tensor))

