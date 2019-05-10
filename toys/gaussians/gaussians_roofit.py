# import ROOT
from collections import defaultdict

import ROOT
from ROOT import RooFit
import progressbar
import yaml
import numpy as np

import zfit_benchmark


def toy_run(n_params, n_gauss, n_toys, toys_nevents, run_zfit, intermediate_result_factory=None):
    # pdf = chebys[0]

    # zfit.settings.set_verbosity(10)

    performance = {}
    performance["column"] = "number of events"
    for nevents in toys_nevents:
        initial_param_val, obs, pdf = build_pdf(n_gauss, n_params, run_zfit)

        # Create dictionary to save fit results
        failed_fits = 0
        successful_fits = 0
        performance[nevents] = {"success": [], "fail": []}

        timer = zfit_benchmark.timer.Timer(f"Toys {nevents}")
        #            mgr = ROOT.RooMCStudy(pdf, obs)
        with progressbar.ProgressBar(max_value=n_toys) as bar:
            ident = 0
            with timer:
                while successful_fits < n_toys:
                    # print(f"starting run number {len(fitResults)}")

                    with timer.child(f"toy number {successful_fits} {ident}") as child:
                        ident += 1
                        data = pdf.generate(ROOT.RooArgSet(obs), int(nevents))
                        pdf.fitTo(data)
                        performance[nevents]["success"].append(float(child.elapsed))
                        bar.update(successful_fits)
                    # mgr.generateAndFit(n_toys, nevents)

        with open("tmp_results.yaml", "w") as f:
            if intermediate_result_factory:
                dump_result = intermediate_result_factory(performance)
            else:
                dump_result = performance.copy()
            dump_result["ATTENTION"] = "NOT FINISHED"
            yaml.dump(dump_result, f)
    return performance


def build_pdf(n_gauss, n_params, run_zfit):
    lower = -1
    upper = 1
    # create observables
    obs = ROOT.RooRealVar("obs", "obs1", lower, upper)
    # create parameters
    params = []
    for i in range(n_params):
        mu = ROOT.RooRealVar(f"mu_{i}", "Mean of Gaussian", -10, 10)
        sigma = ROOT.RooRealVar(f"sigma_{i}", "Width of Gaussian", 3, -10, 10)
    params.append((mu, sigma))
    # create pdfs
    pdfs = []
    for i in range(n_gauss):
        mu, sigma = params[i % n_params]
        shift1 = RooFit.RooConst(float(0.3 * i))
        shifted_mu = ROOT.RooAddition("mu_shifted_{i}", f"Shifted mu {i}", ROOT.RooArgList(mu, shift1))
        shift2 = RooFit.RooConst(float(0.1 * i))
        shifted_sigma = ROOT.RooAddition("sigma_shifted_{i}", f"Shifted sigma {i}", ROOT.RooArgList(sigma, shift2))
        pdf = ROOT.RooGaussian("pdf", "Gaussian pdf", obs, shifted_mu, shifted_sigma)
        pdfs.append(pdf)
    initial_param_val = 1 / n_gauss
    fracs = []
    for i in range(n_gauss - 1):
        frac_value = 1 / n_gauss
        lower_value = 0.0001
        upper_value = 1.5 / n_gauss
        frac = ROOT.RooRealVar("frac", "Fraction of a gauss", frac_value, lower_value, upper_value)
        fracs.append(frac)
    sum_pdf = ROOT.RooAddPdf("sum_pdf", "sum of pdfs", ROOT.RooArgList(*pdfs), ROOT.RooArgList(*fracs))
    pdf = sum_pdf
    return initial_param_val, obs, pdf


if __name__ == '__main__':

    # testing = False
    testing = True
    # run_zfit = False
    run_zfit = False
    n_gauss_max = 50
    n_params_max = n_gauss_max
    toys_nevents = [2 ** i for i in range(7, 24, 2)]
    n_toys = 25

    if testing:
        n_gauss_max = 2
        toys_nevents = [100000]
        n_toys = 100
    results = {}
    results["n_toys"] = n_toys
    results["column"] = "number of gaussians"
    just_one = 0
    for n_gauss in list(range(2, 6)) + list(range(6, 12, 2)) + list(range(12, n_gauss_max + 1, 4)):

        if n_gauss > n_gauss_max:
            break
        results[n_gauss] = {}
        results[n_gauss]["column"] = "number of free params"
        for n_params in range(1, n_gauss + 1):
            # HACK START
            # if just_one > 0:
            #     break
            # just_one += 1
            # HACK END
            if n_gauss < n_gauss_max and n_params not in (1, n_gauss):
                continue  # only test the parameter scan for full params
            results_copy = results.copy()


            def intermediate_result_factory(res_tmp):
                results_copy[n_gauss][n_params] = res_tmp
                return results_copy


            # with tf.device("/device:GPU:0"):
            results[n_gauss][n_params] = toy_run(n_params=n_params, n_gauss=n_gauss,
                                                 n_toys=n_toys, toys_nevents=toys_nevents,
                                                 run_zfit=run_zfit,
                                                 intermediate_result_factory=intermediate_result_factory)

    with open(f"result_{np.random.randint(low=0, high=int(1e18))}.yaml", "w") as f:
        yaml.dump(results, f)
