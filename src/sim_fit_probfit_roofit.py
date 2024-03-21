#!/usr/bin/env python
# coding: utf-8

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
from iminuit import Minuit
# from probfit import UnbinnedLH, gaussian, AddPdf, Normalized, Extended, describe, gen_toy, rename, SimultaneousFit
import time

results = {'probfit': [26],
           'zfit_eager': [6, 10, 12],
           'zfit': [1.6 + 0.5, 3 + 1.5, 4 + 1.5, 11],
           'roofit': [0.5, 2, 4, 12],
           'nevents': [10000, 50000, 100000, 300000]
           }

# In[2]:
do_probfit = False
# do_probfit = False
# zfit_eager = True
zfit_eager = False

nevents = 150000


def gen_samples(nevents, fraction=0.9, slope=0.005):
    fract = sum(np.random.binomial(1, 1 - fraction, nevents) == 0)
    bound = (2900, 3300)
    bkg_m = gen_toy(lambda x: slope * np.math.exp(-slope * x), nevents // 2, bound)
    sig_m = np.random.normal(3096.916, 12, fract)
    tot_m = np.concatenate([sig_m, bkg_m])

    bkg_u = gen_toy(lambda x: slope * np.math.exp(-slope * x), nevents // 2, bound)
    sig_u = np.random.normal(3096.916, 12, nevents - fract)
    tot_u = np.concatenate([sig_u, bkg_u])

    print("matching efficiency = ", fract / nevents)

    return tot_m, tot_u


tot_m, tot_u = gen_samples(nevents=nevents)


# def exp(x, l):
#     return l * np.exp(-l * x)


# def model(fit_range, bin):
#     nrm_bkg_pdf = Normalized(rename(exp, ['x', 'l%d' % bin]), fit_range)
#     ext_bkg_pdf = Extended(nrm_bkg_pdf, extname='Ncomb_%d' % bin)

#     ext_sig_pdf = Extended(rename(gaussian, ['x', 'm%d' % bin, "sigma%d" % bin]), extname='Nsig_%d' % bin)
#     tot_pdf = AddPdf(ext_bkg_pdf, ext_sig_pdf)
#     print('pdf: {}'.format(describe(tot_pdf)))

#     return tot_pdf


# fit_range = (2900, 3300)

# mod_1 = model(fit_range, 1)
# lik_1 = UnbinnedLH(mod_1, tot_m, extended=True)
# mod_2 = model(fit_range, 2)
# lik_2 = UnbinnedLH(mod_2, tot_u, extended=True)
# sim_lik = SimultaneousFit(lik_1, lik_2)
# describe(sim_lik)

# pars = dict(l1=0.002, Ncomb_1=1000, m1=3100, sigma1=10, Nsig_1=1000, l2=0.002, Ncomb_2=1000, m2=3100, sigma2=10,
#             Nsig_2=1000)
# minuit = Minuit(sim_lik, pedantic=False, print_level=0, **pars)

# # In[8]:


# if do_probfit:
#     start = time.time()
#     minuit.migrad()
#     time_probfit = time.time() - start

print("starting zfit")
import zfit

zfit.run.set_graph_mode(not zfit_eager)

mass = zfit.Space("mass", limits=fit_range)


def zfit_model(obs, bin, limits):
    mu = zfit.Parameter("mu{}".format(bin), 3100, limits[0], limits[1])
    sigma = zfit.Parameter("sigma{}".format(bin), 10, 1, 30)
    gauss = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs)

    slope = zfit.Parameter("slope{}".format(bin), -0.002, -0.05, 0.0)
    exp = zfit.pdf.Exponential(lambda_=slope, obs=obs)

    Nsig = zfit.Parameter("Nsig{}".format(bin), 1000, 0, 1000000)
    Nbkg = zfit.Parameter("Nbkg{}".format(bin), 1000, 0, 2000000)
    ext_gauss = gauss.create_extended(Nsig)
    ext_exp = exp.create_extended(Nbkg)

    model = zfit.pdf.SumPDF([ext_exp, ext_gauss])
    return model


model_ = [zfit_model(mass, i, fit_range) for i in range(2)]
data_ = [zfit.Data.from_numpy(obs=mass.obs, array=mass.filter(dataset)) for dataset in [tot_m, tot_u]]

nll_simultaneous = zfit.loss.ExtendedUnbinnedNLL(model=model_, data=data_)
minimizer = zfit.minimize.Minuit(ncall=10000, verbosity=7, tolerance=1e-3, use_minuit_grad=False)

start = time.time()
nll_simultaneous.value_gradients(params=list(nll_simultaneous.get_params()))
time_zfit_compile = time.time() - start

start = time.time()
result = minimizer.minimize(nll_simultaneous)
time_zfit_min = time.time() - start
print(result)
print(result.params)

# x = tf.linspace(mass.lower[0][0], mass.upper[0][0], num=1000)
# nbins = 40
# for mod, data in zip(model_, [tot_m, tot_u]):
#     y = mod.pdf(x) * mod.get_yield() / nbins * mass.rect_area()
#     plt.figure()
#     plt.plot(x, y, label=mod.name)
#     plt.hist(data, bins=nbins)

# Test ROOT too
from ROOT import RooDataSet, RooRealVar, RooGaussian, RooExponential, RooAbsRealLValue, \
    RooArgSet, RooFit, RooCategory, RooSimultaneous, RooArgList, RooAddPdf


def load_set(array, var, dataset):
    for entry in array:
        RooAbsRealLValue.__assign__(var, entry)
        dataset.add(RooArgSet(var))
    return dataset


m = RooRealVar("Jpsi_M", "mass", fit_range[0], fit_range[1])
data_m = RooDataSet("data_m", "data_m", RooArgSet(m))
data_u = RooDataSet("data_u", "data_u", RooArgSet(m))

data_m = load_set(tot_m, m, data_m)
data_u = load_set(tot_u, m, data_u)

data_m.Print("v")
data_u.Print("v")

sample = RooCategory("sample", "sample")
sample.defineType("matched")
sample.defineType("unmatched")

# define the combined set
combData = RooDataSet(
    "combData",
    "combined data",
    RooArgSet(m),
    RooFit.Index(sample),
    RooFit.Import(
        "matched",
        data_m),
    RooFit.Import(
        "unmatched",
        data_u))
combData.Print("v")

# Not working below, bug?
# # create model
# def model(var, bin):
#     # define signal pdf
#     mean = RooRealVar("mean{}".format(bin), "mean{}".format(bin), 3090, 2900, 3300)
#     sigma = RooRealVar("sigma{}".format(bin), "sigma{}".format(bin), 10, 0, 30)
#     gaus = RooGaussian("gx{}".format(bin), "gx{}".format(bin), var, mean, sigma)
#
#     # define background pdf
#     slope = RooRealVar("slope{}".format(bin), "slope{}".format(bin), -0.04, -0.1, -0.0001)
#     exp = RooExponential("exp{}".format(bin), "exp{}".format(bin), var, slope)
#
#     # define yields
#     nsig = RooRealVar("nsig{}".format(bin), "n. sig bin{}".format(bin), 1000, 0., 1000000)
#     nbkg = RooRealVar("nbkg{}".format(bin), "n. bkg bin{}".format(bin), 1000, 0, 2000000)
#
#     # sum pdfs
#     model = RooAddPdf("model{}".format(bin), "model{}".format(bin),
#                       RooArgList(exp, gaus),
#                       RooArgList(nbkg, nsig))
#     return model


# define signal pdf
bin = "0"
mean0 = RooRealVar("mean{}".format(bin), "mean{}".format(bin), 3090, 2900, 3300)
sigma0 = RooRealVar("sigma{}".format(bin), "sigma{}".format(bin), 10, 0, 30)
gaus0 = RooGaussian("gx{}".format(bin), "gx{}".format(bin), m, mean0, sigma0)

# define background pdf
slope0 = RooRealVar("slope{}".format(bin), "slope{}".format(bin), -0.005, -0.1, -0.0001)
exp0 = RooExponential("exp{}".format(bin), "exp{}".format(bin), m, slope0)

# define yields
nsig0 = RooRealVar("nsig{}".format(bin), "n. sig bin{}".format(bin), 1000, 0., 1000000)
nbkg0 = RooRealVar("nbkg{}".format(bin), "n. bkg bin{}".format(bin), 1000, 0, 2000000)

# sum pdfs
model0 = RooAddPdf("model{}".format(bin), "model{}".format(bin),
                   RooArgList(exp0, gaus0),
                   RooArgList(nbkg0, nsig0))

# define signal pdf
bin = "1"
mean1 = RooRealVar("mean{}".format(bin), "mean{}".format(bin), 3090, 2900, 3300)
sigma1 = RooRealVar("sigma{}".format(bin), "sigma{}".format(bin), 10, 0, 30)
gaus1 = RooGaussian("gx{}".format(bin), "gx{}".format(bin), m, mean1, sigma1)

# define background pdf
slope1 = RooRealVar("slope{}".format(bin), "slope{}".format(bin), -0.005, -0.01, -0.0001)
exp1 = RooExponential("exp{}".format(bin), "exp{}".format(bin), m, slope1)

# define yields
nsig1 = RooRealVar("nsig{}".format(bin), "n. sig bin{}".format(bin), 1000, 0., 1000000)
nbkg1 = RooRealVar("nbkg{}".format(bin), "n. bkg bin{}".format(bin), 1000, 0, 2000000)

# sum pdfs
model1 = RooAddPdf("model{}".format(bin), "model{}".format(bin),
                   RooArgList(exp1, gaus1),
                   RooArgList(nbkg1, nsig1))

simPdf = RooSimultaneous("simPdf", "simultaneous pdf", sample)
simPdf.addPdf(model0, "matched")
simPdf.addPdf(model1, "unmatched")

start = time.time()
result = simPdf.fitTo(combData, RooFit.Save(True), RooFit.NumCPU(12))
time_roofit = time.time() - start

if do_probfit:
    print(f"time probfit: {time_probfit}")
print(f"time RooFit: {time_roofit}")
print(f"time zfit {'eager' if zfit_eager else 'graph'} compile: {time_zfit_compile}, time zfit min={time_zfit_min}")
