# import ROOT
from collections import defaultdict

import ROOT
from ROOT import RooRealVar, RooGaussian, RooAddPdf, RooArgList, RooArgSet
from ROOT import RooFit
import progressbar
import yaml
import numpy as np

import zfit_benchmark


def toy_run(nevents):
    lower = -1
    upper = 1
    # create observables
    obs = RooRealVar("obs", "obs1", lower, upper)
    # create parameters
    mean1 = RooRealVar("mean1", "mean of gaussian", 0, -1, 1)
    sigma1 = RooRealVar("sigma1", "sigma of gaussian", 0.1, -1, 1)
    gauss1 = RooGaussian("gauss1", "gaussian PDF", obs, mean1, sigma1)

    mean2 = RooRealVar("mean2", "mean of gaussian", 0.5, -1, 1)
    sigma2 = RooRealVar("sigma2", "sigma of gaussian", 0.2, -1, 1)
    gauss2 = RooGaussian("gauss2", "gaussian PDF", obs, mean2, sigma2)
    frac = RooRealVar("frac", "Fraction of a gauss", 0.5, 0, 1)
    arg_list = RooArgList(gauss1, gauss2, gauss2, gauss2, gauss2,
                          # gauss2,
                          gauss2, gauss2, gauss1)
    arg_list.addOwned(gauss2)
    pdf = RooAddPdf("sum_pdf", "sum of pdfs", arg_list,
                    RooArgList(frac,
                               frac,
                               frac,
                               # frac,
                               # frac,
                               frac,
                               frac,
                               frac,
                               frac,
                               frac))


    # obs, pdf = build_pdf()

    timer = zfit_benchmark.timer.Timer(f"Toys {nevents}")
    with timer:
        data = pdf.generate(RooArgSet(obs), nevents)
        pdf.fitTo(data)
        # mgr.generateAndFit(n_toys, nevents)

    return float(timer.elapsed)


def build_pdf():
    lower = -1
    upper = 1
    # create observables
    obs = RooRealVar("obs", "obs1", lower, upper)
    # create parameters
    mean1 = RooRealVar("mean1", "mean of gaussian", 0, -1, 1)
    sigma1 = RooRealVar("sigma1", "sigma of gaussian", 0.1, -1, 1)
    gauss1 = RooGaussian("gauss1", "gaussian PDF", obs, mean1, sigma1)

    mean2 = RooRealVar("mean2", "mean of gaussian", 0.5, -1, 1)
    sigma2 = RooRealVar("sigma2", "sigma of gaussian", 0.2, -1, 1)
    gauss2 = RooGaussian("gauss2", "gaussian PDF", obs, mean2, sigma2)


    mean3 = RooRealVar("mean3", "mean of gaussian", 0.5, -1, 1)
    sigma3 = RooRealVar("sigma3", "sigma of gaussian", 0.3, -1, 1)
    gauss3 = RooGaussian("gauss3", "gaussian PDF", obs, mean3, sigma3)


    mean4 = RooRealVar("mean4", "mean of gaussian", 0.5, -1, 1)
    sigma4 = RooRealVar("sigma4", "sigma of gaussian", 0.4, -1, 1)
    gauss4 = RooGaussian("gauss4", "gaussian PDF", obs, mean4, sigma4)


    mean5 = RooRealVar("mean5", "mean of gaussian", 0.5, -1, 1)
    sigma5 = RooRealVar("sigma5", "sigma of gaussian", 0.5, -1, 1)
    gauss5 = RooGaussian("gauss5", "gaussian PDF", obs, mean5, sigma5)

    frac1 = RooRealVar("frac", "Fraction of a gauss", 0.5, 0, 1)
    frac2 = RooRealVar("frac", "Fraction of a gauss", 0.5, 0, 1)
    frac3 = RooRealVar("frac", "Fraction of a gauss", 0.5, 0, 1)
    frac4 = RooRealVar("frac", "Fraction of a gauss", 0.5, 0, 1)
    model = RooAddPdf("sum_pdf", "sum of pdfs", RooArgList(RooArgList(gauss1, gauss2),
                                                           RooArgList(gauss3, gauss4, gauss5)),
                      RooArgList(frac1, frac2, frac3, frac4))
    return obs, model


if __name__ == '__main__':

    elapsed = toy_run(nevents=1000)
    print(elapsed)

