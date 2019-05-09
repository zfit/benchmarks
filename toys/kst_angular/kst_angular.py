#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# @file   B2KstLL.py
# @author Albert Puig (albert.puig@cern.ch)
# @date   11.04.2019
# =============================================================================
"""B -> K*ll angular distribution in zfit."""

import argparse
from collections import defaultdict
from math import pi

from typing import Type

import tensorflow as tf
import numpy as np
import pandas as pd
import yaml

import zfit

import progressbar

import matplotlib

matplotlib.use('TkAgg')
# Hack End
import matplotlib.pyplot as plt

import flavio

ztf = zfit.ztf
ztyping = zfit.util.ztyping
ztypes = zfit.settings.ztypes


def plotToys(fitResults):
    """Plotting fit results for the distribution, error and pulls
    for each parameter of the fit
    """

    # Create disctionary with all parameteres
    dictParams = {}
    for key in fitResults[0]:
        dictParams[key.name + '_Val'], dictParams[key.name + '_Err'] = [], []

    # Fill dictionary with all the fitted values
    print(fitResults)
    for iToy in fitResults:
        for i, (key, listpar) in enumerate(iToy.items()):
            dictParams[key.name + '_Val'].append(list(listpar.values())[0])
            dictParams[key.name + '_Err'].append(list(list(listpar.values())[1].values())[0])

    for key, par in dictParams.items():
        # LHCb style
        # matplotlib.rc_file('/Users/rsilvaco/Research/PosDoc/Packages/zfit/zfit-tutorials/LHCb_Style.mlpstyle')
        _par = np.array(par)
        # print(np.mean(_par))
        # print(np.std(_par))
        plt.hist(_par, bins=50)
        plt.savefig("plots/" + key + ".png")
        plt.clf()


def _setInitVal(dictParams, pred, lepton, _q2min, _q2max):
    channel = "B0->K*ee"
    if (lepton): channel = "B0->K*mumu"

    for key, par in dictParams.items():
        if (pred == 'NP'):
            wc = flavio.WilsonCoefficients()
            if (key == 'AT2'):
                par.set_value(
                    flavio.np_prediction('<P1>(' + channel + ')', wc, q2min=float(_q2min), q2max=float(_q2max)))
            else:
                par.set_value(flavio.np_prediction('<' + key + '>(' + channel + ')', wc, q2min=float(_q2min),
                                                   q2max=float(_q2max)))
        else:
            if (key == 'AT2'):
                par.set_value(flavio.sm_prediction('<P1>(' + channel + ')', q2min=float(_q2min), q2max=float(_q2max)))
            else:
                par.set_value(
                    flavio.sm_prediction('<' + key + '>(' + channel + ')', q2min=float(_q2min), q2max=float(_q2max)))

            # The PDFs


class P4pPDF(zfit.pdf.ZPDF):
    """P4prime observable from Bd -> Kst ll (l=e,mu).

    Angular distribution obtained from a fold tecnhique,
        i.e. the valid of the angles is given for
            - phi: [0, pi]
            - theta_K: [0, pi]
            - theta_l: [0, pi/2]

        The function is normalized over a finite range and therefore a PDF.

        Args:

            FL (`zfit.Parameter`): Fraction of longitudinal polarisation of the Kst
            AT2 (`zfit.Parameter`): Transverse asymmetry
            P4p (`zfit.Parameter`): Defined as S4/sqrt(FL(1-FL))
            obs (`zfit.Space`):
            name (str):
            dtype (tf.DType):
    """
    _PARAMS = ['FL', 'AT2', 'P4p']
    _N_OBS = 3

    def _unnormalized_pdf(self, x):
        FL = self.params['FL']
        AT2 = self.params['AT2']
        P4p = self.params['P4p']
        costheta_k, costheta_l, phi = ztf.unstack_x(x)

        sintheta_k = tf.sqrt(1.0 - costheta_k * costheta_k)
        sintheta_l = tf.sqrt(1.0 - costheta_l * costheta_l)

        sintheta_2k = (1.0 - costheta_k * costheta_k)
        sintheta_2l = (1.0 - costheta_l * costheta_l)

        sin2theta_k = (2.0 * sintheta_k * costheta_k)
        cos2theta_l = (2.0 * costheta_l * costheta_l - 1.0)

        pdf = (3.0 / 4.0) * (1.0 - FL) * sintheta_2k + \
              FL * costheta_k * costheta_k + \
              (1.0 / 4.0) * (1.0 - FL) * sintheta_2k * cos2theta_l + \
              -1.0 * FL * costheta_k * costheta_k * cos2theta_l + \
              (1.0 / 2.0) * (1.0 - FL) * AT2 * sintheta_2k * sintheta_2l * tf.cos(2.0 * phi) + \
              tf.sqrt(FL * (1 - FL)) * P4p * sin2theta_k * sin2theta_l * tf.cos(phi)

        return pdf


class P5pPDF(zfit.pdf.ZPDF):
    _PARAMS = ['FL', 'AT2', 'P5p']
    _N_OBS = 3

    def _unnormalized_pdf(self, x):
        FL = self.params['FL']
        AT2 = self.params['AT2']
        P5p = self.params['P5p']
        costheta_k, costheta_l, phi = ztf.unstack_x(x)

        sintheta_k = tf.sqrt(1.0 - costheta_k * costheta_k)
        sintheta_l = tf.sqrt(1.0 - costheta_l * costheta_l)

        sintheta_2k = (1.0 - costheta_k * costheta_k)
        sintheta_2l = (1.0 - costheta_l * costheta_l)

        sin2theta_k = (2.0 * sintheta_k * costheta_k)
        cos2theta_l = (2.0 * costheta_l * costheta_l - 1.0)

        pdf = (3.0 / 4.0) * (1.0 - FL) * sintheta_2k + \
              FL * costheta_k * costheta_k + \
              (1.0 / 4.0) * (1.0 - FL) * sintheta_2k * cos2theta_l + \
              -1.0 * FL * costheta_k * costheta_k * cos2theta_l + \
              (1.0 / 2.0) * (1.0 - FL) * AT2 * sintheta_2k * sintheta_2l * tf.cos(2.0 * phi) + \
              tf.sqrt(FL * (1 - FL)) * P5p * sin2theta_k * sintheta_l * tf.cos(phi)

        return pdf


class P6pPDF(zfit.pdf.ZPDF):
    """P6prime observable from Bd -> Kst ll (l=e,mu).

    Angular distribution obtained from a fold tecnhique,
        i.e. the valid of the angles is given for
            - phi: [-pi/2, pi/2]
            - theta_K: [0, pi]
            - theta_l: [0, pi/2]

        The function is normalized over a finite range and therefore a PDF.

        Args:

            FL (`zfit.Parameter`): Fraction of longitudinal polarisation of the Kst
            AT2 (`zfit.Parameter`): Transverse asymmetry
            P6p (`zfit.Parameter`): Defined as S5/sqrt(FL(1-FL))
            obs (`zfit.Space`):
            name (str):
            dtype (tf.DType):

    """
    _PARAMS = ['FL', 'AT2', 'P6p']
    _N_OBS = 3

    def _unnormalized_pdf(self, x):
        FL = self.params['FL']
        AT2 = self.params['AT2']
        P6p = self.params['P6p']
        costheta_k, costheta_l, phi = ztf.unstack_x(x)

        sintheta_k = tf.sqrt(1.0 - costheta_k * costheta_k)
        sintheta_l = tf.sqrt(1.0 - costheta_l * costheta_l)

        sintheta_2k = (1.0 - costheta_k * costheta_k)
        sintheta_2l = (1.0 - costheta_l * costheta_l)

        sin2theta_k = (2.0 * sintheta_k * costheta_k)
        cos2theta_l = (2.0 * costheta_l * costheta_l - 1.0)

        pdf = (3.0 / 4.0) * (1.0 - FL) * sintheta_2k + \
              FL * costheta_k * costheta_k + \
              (1.0 / 4.0) * (1.0 - FL) * sintheta_2k * cos2theta_l + \
              -1.0 * FL * costheta_k * costheta_k * cos2theta_l + \
              (1.0 / 2.0) * (1.0 - FL) * AT2 * sintheta_2k * sintheta_2l * tf.cos(2.0 * phi) + \
              tf.sqrt(FL * (1 - FL)) * P6p * sin2theta_k * sintheta_l * tf.sin(phi)

        return pdf


class P8pPDF(zfit.pdf.ZPDF):
    """P8prime observable from Bd -> Kst ll (l=e,mu).

    Angular distribution obtained from a fold tecnhique,
        i.e. the valid of the angles is given for
            - phi: [-pi/2, pi/2]
            - theta_K: [0, pi]
            - theta_l: [0, pi/2]

        The function is normalized over a finite range and therefore a PDF.

        Args:

            FL (`zfit.Parameter`): Fraction of longitudinal polarisation of the Kst
            AT2 (`zfit.Parameter`): Transverse asymmetry
            P8p (`zfit.Parameter`): Defined as S5/sqrt(FL(1-FL))
            obs (`zfit.Space`):
            name (str):
            dtype (tf.DType):

    """
    _PARAMS = ['FL', 'AT2', 'P8p']
    _N_OBS = 3

    def _unnormalized_pdf(self, x):
        FL = self.params['FL']
        AT2 = self.params['AT2']
        P8p = self.params['P8p']
        costheta_k, costheta_l, phi = ztf.unstack_x(x)

        sintheta_k = tf.sqrt(1.0 - costheta_k * costheta_k)
        sintheta_l = tf.sqrt(1.0 - costheta_l * costheta_l)

        sintheta_2k = (1.0 - costheta_k * costheta_k)
        sintheta_2l = (1.0 - costheta_l * costheta_l)

        sin2theta_k = (2.0 * sintheta_k * costheta_k)
        cos2theta_l = (2.0 * costheta_l * costheta_l - 1.0)

        pdf = (3.0 / 4.0) * (1.0 - FL) * sintheta_2k + \
              FL * costheta_k * costheta_k + \
              (1.0 / 4.0) * (1.0 - FL) * sintheta_2k * cos2theta_l + \
              -1.0 * FL * costheta_k * costheta_k * cos2theta_l + \
              (1.0 / 2.0) * (1.0 - FL) * AT2 * sintheta_2k * sintheta_2l * tf.cos(2.0 * phi) + \
              tf.sqrt(FL * (1 - FL)) * P8p * sin2theta_k * sin2theta_l * tf.sin(phi)

        return pdf


# Folding data

def fold_P4p(data, costheta_k, costheta_l, phi):
    theta_l = np.acos(data[costheta_l])

    data[f'{costheta_k}_P4p'] = data[costheta_k]
    data[f'{phi}_P4p'] = np.where(data[phi] < 0,
                                  -data[phi],
                                  data[phi])
    data[f'{phi}_P4p'] = np.where(theta_l > 0.5 * pi,
                                  pi - data[f'{phi}_P4p'],
                                  data[f'{phi}_P4p'])
    data[f'{costheta_l}_P4p'] = np.where(theta_l > 0.5 * pi,
                                         np.cos(pi - theta_l),
                                         data[costheta_l])

    return zfit.data.Data.from_pandas(data[f'{costheta_l}_P4p',
                                           f'{costheta_k}_P4p',
                                           f'{phi}_P4p'].copy()
                                      .rename(index=str,
                                              columns={f'{costheta_l}_P4p': costheta_l,
                                                       f'{costheta_k}_P4p': costheta_k,
                                                       f'{phi}_P4p': phi}))


def fold_P5p(data, costheta_k, costheta_l, phi):
    theta_l = np.acos(data[costheta_l])

    data[f'{costheta_k}_P5p'] = data[costheta_k]
    data[f'{phi}_P5p'] = np.where(data[f'{phi}_P5p'] < 0,
                                  -data[f'{phi}_P5p'],
                                  data[f'{phi}_P5p'])
    data[f'{costheta_l}_P5p'] = np.where(theta_l > 0.5 * pi,
                                         np.cos(pi - theta_l),
                                         data[costheta_l])

    return zfit.data.Data.from_pandas(data[f'{costheta_l}_P5p',
                                           f'{costheta_k}_P5p',
                                           f'{phi}_P5p'].copy()
                                      .rename(index=str,
                                              columns={f'{costheta_l}_P5p': costheta_l,
                                                       f'{costheta_k}_P5p': costheta_k,
                                                       f'{phi}_P5p': phi}))


def fold_P6p(data, costheta_k, costheta_l, phi):
    theta_l = np.acos(data[costheta_l])

    data[f'{costheta_k}_P6p'] = data[costheta_k]
    data[f'{phi}_P6p'] = np.where(data[phi] > 0.5 * pi,
                                  pi - data[phi],
                                  data[phi])
    data[f'{phi}_P6p'] = np.where(data[f'{phi}_P6p'] < - 0.5 * pi,
                                  - pi - data[f'{phi}_P6p'],
                                  data[f'{phi}_P6p'])
    data[f'{costheta_l}_P6p'] = np.where(theta_l > 0.5 * pi,
                                         np.cos(pi - theta_l),
                                         data[costheta_l])

    return zfit.data.Data.from_pandas(data[f'{costheta_l}_P6p',
                                           f'{costheta_k}_P6p',
                                           f'{phi}_P6p'].copy()
                                      .rename(index=str,
                                              columns={f'{costheta_l}_P6p': costheta_l,
                                                       f'{costheta_k}_P6p': costheta_k,
                                                       f'{phi}_P6p': phi}))


def fold_P8p(data, costheta_k, costheta_l, phi):
    theta_k = np.acos(data[costheta_k])
    theta_l = np.acos(data[costheta_l])

    data[f'{costheta_k}_P8p'] = np.where(theta_l > 0.5 * pi,
                                         np.cos(pi - theta_k),
                                         data[costheta_k])

    data[f'{phi}_P8p'] = np.where(data[phi] > 0.5 * pi,
                                  pi - data[phi],
                                  data[phi])
    data[f'{phi}_P8p'] = np.where(data[f'{phi}_P8p'] < - 0.5 * pi,
                                  - pi - data[f'{phi}_P8p'],
                                  data[f'{phi}_P8p'])
    data[f'{costheta_l}_P8p'] = np.where(theta_l > 0.5 * pi,
                                         np.cos(pi - theta_l),
                                         data[costheta_l])

    return zfit.data.Data.from_pandas(data[f'{costheta_l}_P8p',
                                           f'{costheta_k}_P8p',
                                           f'{phi}_P8p'].copy()
                                      .rename(index=str,
                                              columns={f'{costheta_l}_P8p': costheta_l,
                                                       f'{costheta_k}_P8p': costheta_k,
                                                       f'{phi}_P8p': phi}))


# A bit of handling

class B2Kstll:
    FOLDS = {'P4p': (P4pPDF, ['FL', 'AT2', 'P4p'], fold_P4p),
             'P5p': (P5pPDF, ['FL', 'AT2', 'P5p'], fold_P5p),
             'P6p': (P6pPDF, ['FL', 'AT2', 'P6p'], fold_P6p),
             'P8p': (P8pPDF, ['FL', 'AT2', 'P8p'], fold_P8p)}

    def __init__(self, costheta_l, costheta_k, phi):
        self._obs_names = {'costheta_l': costheta_l.obs,
                           'costheta_k': costheta_k.obs,
                           'phi': phi.obs}
        self.obs = costheta_l * costheta_k * phi
        self.params = {}

    def get_folded_pdf(self, name):
        pdf_class, param_names, _ = self.FOLDS[name]

        def get_params(param_list):
            out = {}
            for param in param_list:
                if param not in self.params:
                    config = [0.8, 0, 1] if param == 'FL' else [0.0, -1, 1]
                    self.params.update({param: zfit.Parameter(param, *config)})
                out[param] = self.params[param]
            return out

        # Make sure params exist
        params = get_params(param_names)
        pdf = pdf_class(obs=self.obs, **params)
        return pdf

    def fold_dataset(self, name, dataset):
        *_, data_transform = self.FOLDS[name]
        return data_transform(dataset, self.obs.obs)


def run_toys(pdf_factory, n_toys, toys_nevents):
    zfit.run.create_session(reset_graph=True)
    pdf = pdf_factory()
    sampler = pdf.create_sampler(n=1000)
    nll = zfit.loss.UnbinnedNLL(model=pdf, data=sampler, fit_range=pdf.space)
    # minimizer = zfit.minimize.MinuitMinimizer(verbosity=0)
    from zfit.minimizers.baseminimizer import ToyStrategyFail
    minimizer = zfit.minimize.MinuitMinimizer(strategy=ToyStrategyFail(), verbosity=0)

    # pre build graph
    sampler.resample(n=1000)
    zfit.run([nll.value(), nll.gradients()])
    dependents = pdf.get_dependents()
    performance = {}
    performance["ntoys"] = n_toys
    for nevents in toys_nevents:

        # Create dictionary to save fit results

        performance[nevents] = {"success": [], "fail": []}

        failed_fits = 0
        successful_fits = 0

        timer = Timer(f"Toys {nevents}")
        with progressbar.ProgressBar(max_value=n_toys) as bar:
            ident = 0
            with timer:
                while successful_fits < n_toys:
                    with timer.child(f"toy number {successful_fits} {ident}") as child:
                        # Retrieve value from flav.io predictions
                        _setInitVal(pdf.params, pred, lepton, _q2min, _q2max)

                        # Generate toys
                        sampler.resample(n=nevents)

                        # Randomise initial values
                        for param in dependents:
                            param.randomize()

                        # Minimise the NLL
                        minimum = minimizer.minimize(nll)
                    if ident == 0:
                        ident += 1
                        continue
                    if minimum.converged:
                        bar.update(successful_fits)
                        successful_fits += 1
                        fail_or_success = "success"
                    else:
                        failed_fits += 1
                        fail_or_success = "fail"
                    ident += 1
                    performance[nevents][fail_or_success].append(float(child.elapsed))
    print("Failed fits: {}/{}".format(failed_fits, failed_fits + n_toys))
    return performance
    # Plotting fit results
    # plotToys(fitResults)


def pdf_factory():
    # Phase space
    costheta_l = zfit.Space("costhetal", limits=(0, 1.0))
    costheta_k = zfit.Space("costhetaK", limits=(-1.0, 1.0))
    phi = zfit.Space("phi", limits=(0, pi))
    decay = B2Kstll(costheta_l, costheta_k, phi)
    # Define angular pdf
    angularPDF = decay.get_folded_pdf(fold)
    # Create mass pdf
    mu = zfit.Parameter("mu", 5279, 5200, 5400)
    sigma = zfit.Parameter("sigma", 30, 20, 50)
    a0 = zfit.Parameter("a0", 0.9, 0.7, 2)
    a1 = zfit.Parameter("a1", 1.1, 0.9, 2.5)
    n0 = zfit.Parameter("n0", 7, 5, 9)
    n1 = zfit.Parameter("n1", 4, 3, 6)
    mass = zfit.Space("mass", limits=(4900, 5600))
    massPDF = zfit.pdf.DoubleCB(obs=mass, mu=mu, sigma=sigma,
                                alphal=a0, nl=n0, alphar=a1, nr=n1)
    pdf = massPDF * angularPDF
    # pdf = angularPDF
    return pdf


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Toys of Kst angular')

    parser.add_argument("-t", "--testing", dest="testing", action='store_true', help="Set the minimum q2 for the simulation")
    # parser.add_argument("-i", "--q2min", dest="q2min", required=True, help="Set the minimum q2 for the simulation")
    # parser.add_argument("-j", "--q2max", dest="q2max", required=True, help="Set the maximum q2 for the simulation")
    # parser.add_argument("-f", "--fold", dest="fold", required=True,
    #                     help="Choose the fold to be examined (i.e. P4p, P5p, P6p or P8p)")
    # parser.add_argument("-l", "--lepton", dest="lepton", required=False,
    #                     help="Choose the final state (e.g. muon or electron)")
    # parser.add_argument("-p", "--pred", dest="pred", required=True, help="Choose whether SM or NP prediction")
    #
    args = parser.parse_args()

    # Parameters and configuration
    # _q2min = args.q2min
    # _q2max = args.q2max
    # fold = args.fold
    # lepton = args.lepton
    # pred = args.pred
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    zfit.run.sess = sess

    from zfit_benchmark.timer import Timer

    _q2min = 1.1
    _q2max = 6
    fold = "P5p"
    lepton = "muon"
    pred = "sm"
    testing = args.testing
    print(testing)
    if testing:
        toys_nevents = [2 ** i for i in range(7, 9)]
        n_toys = 3
    else:
        toys_nevents = [2 ** i for i in range(7, 20, 2)]
        n_toys = 25

    results = run_toys(pdf_factory=pdf_factory, n_toys=n_toys, toys_nevents=toys_nevents)
    with open(f"results_{np.random.randint(low=0, high=int(1e18))}.yaml", "w") as f:
        yaml.dump(results, f)

# EOFs
