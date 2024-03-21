import matplotlib.pyplot as plt
import numpy as np
import zfit
import zfit.z.numpy as znp
from zfit import z
import tensorflow as tf
import tensorflow_probability as tfp


@z.function
def ipatia2_tfp(x, l, zeta, fb, mu, sigma, n, n2, a, a2):
    sq2pi = znp.sqrt(2.0 * znp.arccos(-1.0))
    sq2pi_inv = 1.0 / sq2pi
    logsq2pi = znp.log(sq2pi)
    log_de_2 = znp.log(2.0)

    def low_x_BK(nu, x):
        return gammafunc(nu) * znp.power(2.0, nu - 1.0) * znp.power(x, -nu)

    def gammafunc(nu):
        # return ROOT.Math.Gamma(nu)
        return znp.exp(tf.math.lgamma(nu))

    def low_x_LnBK(nu, x):
        return znp.log(gammafunc(nu)) + (nu - 1.0) * log_de_2 - nu * znp.log(x)

    def BK(ni, x):
        nu = znp.abs(ni)
        first_cond = tf.logical_and(x < 1.0e-06, nu > 0.0)
        second_cond = tf.logical_and(tf.logical_and(x < 1.0e-04, nu > 0.0), nu < 55)
        third_cond = tf.logical_and(x < 0.1, nu >= 55)
        cond = tf.logical_or(first_cond, tf.logical_or(second_cond, third_cond))
        return znp.where(cond, low_x_BK(nu, x), tfp.math.bessel_kve(nu, x))

        # return ROOT.Math.cyl_bessel_k(nu, x)
        # return scipy.special.kn(nu, x)
        # ROOT:Math:cyl_bessel_k(nu, x)

    def LnBK(ni, x):
        nu = znp.abs(ni)
        first_cond = tf.logical_and(x < 1.0e-06, nu > 0.0)
        second_cond = tf.logical_and(tf.logical_and(x < 1.0e-04, nu > 0.0), nu < 55)
        third_cond = tf.logical_and(x < 0.1, nu >= 55)
        cond = tf.logical_or(first_cond, tf.logical_or(second_cond, third_cond))
        return znp.where(cond, low_x_LnBK(nu, x), znp.log(tfp.math.bessel_kve(nu, x)))

    def LogEval(d, l, alpha, beta, delta):
        # d = x-mu
        # sq2pi = znp.sqrt(2*znp.arccos(-1))
        gamma = alpha  # znp.sqrt(alpha*alpha-beta*beta)
        dg = delta * gamma
        thing = delta * delta + d * d
        logno = l * znp.log(gamma / delta) - logsq2pi - LnBK(l, dg)

        return znp.exp(
            logno + beta * d + (0.5 - l) * (znp.log(alpha) - 0.5 * znp.log(thing)) + LnBK(l - 0.5, alpha * znp.sqrt(thing))
        )  # + znp.log(znp.abs(beta)+0.0001) )

    def diff_eval(d, l, alpha, beta, delta):
        # sq2pi = znp.sqrt(2*znp.arccos(-1))
        # cons1 = 1./sq2pi
        gamma = alpha  # znp.sqrt(alpha*alpha-beta*beta)
        dg = delta * gamma
        # mu_ = mu# - delta*beta*BK(l+1,dg)/(gamma*BK(l,dg))
        # d = x-mu
        thing = delta * delta + d * d
        sqthing = znp.sqrt(thing)
        alphasq = alpha * sqthing
        no = znp.power(gamma / delta, l) / BK(l, dg) * sq2pi_inv
        ns1 = 0.5 - l

        return (
                no
                * znp.power(alpha, ns1)
                * znp.power(thing, l / 2.0 - 1.25)
                * (-d * alphasq * (BK(l - 1.5, alphasq) + BK(l + 0.5, alphasq)) + (2.0 * (beta * thing + d * l) - d) * BK(ns1, alphasq))
                * znp.exp(beta * d)
                / 2.0
        )

    def Gauss2F1(a, b, c, x):
        largey = tfp.math.hypergeometric.hyp2f1_small_argument(c - a, b, c, 1 - 1 / (1 - x)) / znp.power(1 - x, b)
        smally = tfp.math.hypergeometric.hyp2f1_small_argument(a, b, c, x)
        return znp.where(znp.abs(x) <= 1, smally, largey)
        # if (znp.abs(x) <= 1):
        # return ROOT.Math.hyperg(a, b, c, x)

        # ROOT::Math::hyperg(a,b,c,x)
        # else:
        # return ROOT.Math.hyperg(c - a, b, c, 1 - 1 / (1 - x)) / znp.power(1 - x, b)
        # return largey

    def stIntegral(d1, delta, l):
        # printf("::::: %e %e %e\n", d1,delta, l)
        return d1 * Gauss2F1(0.5, 0.5 - l, 3.0 / 2, -d1 * d1 / (delta * delta))
        # printf(":::Done\n")
        # return out

    def ipatia2(x, l, zeta, fb, mu, sigma, n, n2, a, a2):
        d = x - mu
        cons0 = znp.sqrt(zeta)
        asigma = a * sigma
        a2sigma = a2 * sigma
        cond1 = d < -asigma
        cond2 = d > a2sigma
        conda1 = zeta != 0.0
        conda2 = l < 0.0
        # cond1
        phi = BK(l + 1.0, zeta) / BK(l, zeta)
        cons1 = sigma / znp.sqrt(phi)
        alpha = cons0 / cons1  # *znp.sqrt((1 - fb*fb))
        beta = fb  # *alpha
        delta = cons0 * cons1

        # printf("-_-\n")
        # printf("alpha %e\n",alpha)
        # printf("beta %e\n",beta)
        # printf("delta %e\n",delta)

        k1 = LogEval(-asigma, l, alpha, beta, delta)
        k2 = diff_eval(-asigma, l, alpha, beta, delta)
        B = -asigma + n * k1 / k2
        A = k1 * znp.power(B + asigma, n)
        out1 = A * znp.power(B - d, -n)

        k1 = LogEval(a2sigma, l, alpha, beta, delta)
        k2 = diff_eval(a2sigma, l, alpha, beta, delta)

        B = -a2sigma - n2 * k1 / k2

        A = k1 * znp.power(B + a2sigma, n2)

        out2 = A * znp.power(B + d, -n2)

        out3 = LogEval(d, l, alpha, beta, delta)
        outa1 = znp.where(cond1, out1, znp.where(cond2, out2, out3))

        # cond2 = d > a2sigma
        beta = fb
        cons1 = -2.0 * l
        # delta = sigma
        condx = l <= -1.0

        delta1 = sigma * znp.sqrt(-2 + cons1)

        # printf("WARNING: zeta ==0 and l > -1 ==> not defined rms. Changing the meaning of sigma, but I keep fitting anyway\n")
        delta2 = sigma
        delta = znp.where(condx, delta1, delta2)

        delta2 = delta * delta
        # cond1
        cons1 = znp.exp(-beta * asigma)
        phi = 1.0 + asigma * asigma / delta2
        k1 = cons1 * znp.power(phi, l - 0.5)
        k2 = beta * k1 - cons1 * (l - 0.5) * znp.power(phi, l - 1.5) * 2 * asigma / delta2
        B = -asigma + n * k1 / k2
        A = k1 * znp.power(B + asigma, n)
        outz1 = A * znp.power(B - d, -n)
        # cond2
        cons1 = znp.exp(beta * a2sigma)
        phi = 1.0 + a2sigma * a2sigma / delta2
        k1 = cons1 * znp.power(phi, l - 0.5)
        k2 = beta * k1 + cons1 * (l - 0.5) * znp.power(phi, l - 1.5) * 2.0 * a2sigma / delta2
        B = -a2sigma - n2 * k1 / k2
        A = k1 * znp.power(B + a2sigma, n2)
        outz2 = A * znp.power(B + d, -n2)
        # cond3
        outz3 = znp.exp(beta * d) * znp.power(1.0 + d * d / delta2, l - 0.5)

        outa2 = znp.where(cond1, outz1, znp.where(cond2, outz2, outz3))

        out = znp.where(conda1, outa1, outa2)

        # printf("result is %e\n",out)
        return out

    return ipatia2(x, l, zeta, fb, mu, sigma, n, n2, a, a2)


class Ipatia2(zfit.pdf.BasePDF):
    def __init__(self, obs, mu, sigma, nl, al, nr, ar, lam, beta, zeta):
        params = {
            "mu": mu,
            "sigma": sigma,
            "nl": nl,
            "al": al,
            "nr": nr,
            "ar": ar,
            "lam": lam,
            "beta": beta,
            "zeta": zeta,
        }
        super().__init__(obs=obs, name="Ipatia2", params=params)

    def _unnormalized_pdf(self, x):
        x = zfit.z.unstack_x(x)
        mu = self.params["mu"]
        sigma = self.params["sigma"]
        nl = self.params["nl"]
        al = self.params["al"]
        nr = self.params["nr"]
        ar = self.params["ar"]
        lam = self.params["lam"]
        beta = self.params["beta"]
        zeta = self.params["zeta"]
        return ipatia2_tfp(x, lam, zeta, beta, mu, sigma, nl, nr, al, ar)


if __name__ == '__main__':
    # TODO: change parameters
    mu = zfit.Parameter("mu", 0.0)
    sigma = zfit.Parameter("sigma", 1.2)
    nl = zfit.Parameter("nl", 1.0)
    al = zfit.Parameter("al", 1.0)
    nr = zfit.Parameter("nr", 1.0)
    ar = zfit.Parameter("ar", 1.0)
    lam = zfit.Parameter("lam", -0.5)
    beta = zfit.Parameter("beta", 1.0)
    zeta = zfit.Parameter("zeta", 1.0)
    obs = zfit.Space("x", limits=(-5, 15))
    dist = Ipatia2(obs=obs, mu=mu, sigma=sigma, nl=nl, al=al, nr=nr, ar=ar, lam=lam, beta=beta, zeta=zeta)

    zfit.settings.set_verbosity(7)
    data = dist.sample(10000)
    plt.hist(data.numpy(), bins=100, density=True)
    plt.show()
    nll = zfit.loss.UnbinnedNLL(model=dist, data=data)
    minimizer = zfit.minimize.Minuit(verbosity=7)
    zfit.param.set_values([mu, sigma], [0.1, 1.1])
    print("Calculating gradient...")
    print(f"Gradient: {nll.gradient()}")
    result = minimizer.minimize(nll)

    print(result)
    result.hesse()
    print(result)

    plt.figure()
    x = np.linspace(-15, 15, 1000)
    plt.plot(x, dist.pdf(x).numpy())
    plt.hist(data.numpy(), bins=100, density=True)
    plt.show()
