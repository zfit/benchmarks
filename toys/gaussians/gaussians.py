# import ROOT
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import pprint
from collections import defaultdict
from decimal import Decimal

try:
    import ROOT
    from ROOT import RooRealVar, RooGaussian, RooChebychev, RooAddPdf, RooArgList, RooArgSet, RooFit, RooAddition
except:
    pass
import progressbar
import yaml
import zfit
import zfit.minimizers.baseminimizer
import numpy as np

import zfit_benchmark

zfit.run.numeric_checks = False
run_name = "gpu_tol1_grad_new"

def toy_run(n_params, n_gauss, n_toys, toys_nevents, run_zfit, intermediate_result_factory=None):
    # pdf = chebys[0]

    # zfit.settings.set_verbosity(10)

    performance = {}
    performance["column"] = "number of events"
    for nevents in toys_nevents:
        # n_toys = 30 if nevents < 50000 else 10

        if run_zfit:
            zfit.run.create_session(reset_graph=True)
            # zfit.sess.close()
            # zfit.sess = tf.Session
        # initial_param_val, obs, pdf = build_pdf(n_gauss, n_params, run_zfit)

        lower = -1
        upper = 1
        # create observables
        if run_zfit:
            obs = zfit.Space("obs1", limits=(lower, upper))
        else:
            obs = RooRealVar("obs", "obs1", lower, upper)
            ROOT.SetOwnership(obs, False)
        # create parameters
        params = []
        params_initial = []
        mu_lower, mu_upper = 1, 3
        sigma_lower, sigma_upper = 0.5, 2
        # step_size = 0.003
        for i in range(n_params):
            if run_zfit:
                mu = zfit.Parameter(f"mu_{i}_{nevents}", np.random.uniform(low=mu_lower, high=mu_upper), mu_lower,
                                    mu_upper,
                                    # step_size=step_size
                                    )
                sigma = zfit.Parameter(f"sigma_{i}_{nevents}", np.random.uniform(low=sigma_lower, high=sigma_upper),
                                       sigma_lower, sigma_upper,
                                       # step_size=step_size
                                       )
            else:
                mu_initial = np.random.uniform(mu_lower, mu_upper)
                mu = RooRealVar(f"mu_{i}_{nevents}", "Mean of Gaussian", mu_initial, mu_lower, mu_upper)
                ROOT.SetOwnership(mu, False)
                sigma_initial = np.random.uniform(mu_lower, mu_upper)
                sigma = RooRealVar(f"sigma_{i}_{nevents}", "Width of Gaussian", sigma_initial, sigma_lower, sigma_upper)
                ROOT.SetOwnership(sigma, False)
                params_initial.append((mu_initial, sigma_initial))
            params.append((mu, sigma))
        # create pdfs
        pdfs = []
        for i in range(n_gauss):
            mu, sigma = params[i % n_params]
            if run_zfit:
                shifted_mu = mu + 0.3 * i
                shifted_sigma = sigma + 0.1 * i
                pdf = zfit.pdf.Gauss(obs=obs, mu=shifted_mu, sigma=shifted_sigma)
                # from zfit.models.basic import CustomGaussOLD
                # pdf = CustomGaussOLD(obs=obs, mu=shifted_mu, sigma=shifted_sigma)
                # pdf.update_integration_options(mc_sampler=tf.random_uniform)
            else:
                shift1 = RooFit.RooConst(float(0.3 * i))
                shifted_mu = RooAddition(f"mu_shifted_{i}_{nevents}", f"Shifted mu {i}", RooArgList(mu, shift1))
                shift2 = RooFit.RooConst(float(0.1 * i))
                shifted_sigma = RooAddition(f"sigma_shifted_{i}_{nevents}", f"Shifted sigma {i}",
                                            RooArgList(sigma, shift2))
                pdf = RooGaussian(f"pdf_{i}_{nevents}", "Gaussian pdf", obs, shifted_mu, shifted_sigma)
                ROOT.SetOwnership(pdf, False)
                ROOT.SetOwnership(shift1, False)
                ROOT.SetOwnership(shifted_mu, False)
                ROOT.SetOwnership(shift2, False)
                ROOT.SetOwnership(shifted_sigma, False)
            pdfs.append(pdf)
        initial_param_val = 1 / n_gauss
        fracs = []
        for i in range(n_gauss - 1):
            frac_value = 1 / n_gauss
            lower_value = 0.0001
            upper_value = 1.5 / n_gauss
            if run_zfit:
                frac = zfit.Parameter(f"frac_{i}", value=1 / n_gauss, lower_limit=lower_value, upper_limit=upper_value)
                frac.floating = False
            else:
                frac = RooRealVar(f"frac_{i}_{nevents}", "Fraction of a gauss", frac_value)
                ROOT.SetOwnership(frac, False)
            fracs.append(frac)
        if run_zfit:
            sum_pdf = zfit.pdf.SumPDF(pdfs=pdfs, fracs=fracs)
            # sum_pdf.update_integration_options(mc_sampler=tf.random_uniform)

        else:
            sum_pdf = RooAddPdf(f"sum_pdf_{nevents}", "sum of pdfs", RooArgList(*pdfs), RooArgList(*fracs))
            ROOT.SetOwnership(sum_pdf, False)
        pdf = sum_pdf

        # Create dictionary to save fit results
        failed_fits = 0
        successful_fits = 0
        performance[nevents] = {"success": [], "fail": []}

        if run_zfit:
            sampler = pdf.create_sampler(n=nevents, fixed_params=True)
            sampler.set_data_range(obs)
            nll = zfit.loss.UnbinnedNLL(pdf, sampler)

            minimizer = zfit.minimize.MinuitMinimizer(zfit.minimizers.baseminimizer.ToyStrategyFail(), verbosity=5,
                                                      minimize_strategy=1)
            # minimizer.minimizer_options['tol'] = 100

            # minimizer._use_tfgrad = False

        timer = zfit_benchmark.timer.Timer(f"Toys {nevents}")
        if run_zfit:
            sampler.resample()
            # with tf.device("/device:GPU:0"):
            jit_scope = tf.contrib.compiler.jit.experimental_jit_scope
            # with jit_scope():
            to_run = [nll.value(), nll.gradients()]
            zfit.run(to_run)
            dependents = pdf.get_dependents()
        else:
            mgr = ROOT.RooMCStudy(pdf, RooArgSet(obs), RooFit.Silence())
            ROOT.SetOwnership(mgr, False)
        run_toystudy = False
        with progressbar.ProgressBar(max_value=n_toys) as bar:
            ident = 0
            with timer:
                if not run_toystudy:
                    while successful_fits < n_toys:
                        # print(f"starting run number {len(fitResults)}")
                        if run_zfit:
                            sampler.resample()

                            for param in dependents:
                                param.randomize()
                        else:
                            for (mu, sigma), (mu_val, sigma_val) in zip(params, params_initial):
                                mu.setVal(mu_val)
                                sigma.setVal(sigma_val)

                            data = pdf.generate(RooArgSet(obs), nevents)

                            for mu, sigma in params:
                                mu.setVal(np.random.uniform(mu_lower, mu_upper))
                                sigma.setVal(np.random.uniform(sigma_lower, sigma_upper))

                        with timer.child(f"toy number {successful_fits} {ident}") as child:
                            if run_zfit:
                                # sampler.resample()

                                # with tf.device("/device:GPU:0"):
                                minimum = minimizer.minimize(nll)
                                # print(minimum.hesse())
                            else:

                                # for mu, sigma in params:
                                #     mu.setVal(np.random.uniform(mu_lower, mu_upper))
                                #     sigma.setVal(np.random.uniform(sigma_lower, sigma_upper))
                                # for frac in fracs:
                                #     frac.setVal(np.random.uniform(lower_value, upper_value))
                                result = pdf.fitTo(data, RooFit.NumCPU(12), RooFit.Save(True),
                                                   RooFit.Hesse(False), RooFit.Minos(False))
                        if ident == 0:
                            ident += 1
                            continue  # warm up run
                        if run_zfit:
                            if minimum.converged:
                                bar.update(successful_fits)
                                successful_fits += 1
                                fail_or_success = "success"
                            else:
                                child.elapsed = Decimal()
                                failed_fits += 1
                                fail_or_success = "fail"
                        else:
                            if result.status() == 0:
                                bar.update(successful_fits)
                                successful_fits += 1
                                fail_or_success = "success"
                            else:
                                child.elapsed = Decimal()
                                failed_fits += 1
                                fail_or_success = "fail"

                        ident += 1
                        performance[nevents][fail_or_success].append(float(child.elapsed))
                else:

                    mgr.generateAndFit(n_toys, nevents)
                    performance[nevents]["success"].append([float(timer.elapsed) / n_toys for _ in range(n_toys)])

        with open(f"{run_name}tmp.yaml", "w") as f:
            if intermediate_result_factory:
                dump_result = intermediate_result_factory(performance)
            else:
                dump_result = performance.copy()
            dump_result["ATTENTION"] = "NOT FINISHED"
            yaml.dump(dump_result, f)
    return performance


if __name__ == '__main__':
    import tensorflow as tf

    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    # sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    config = tf.ConfigProto(intra_op_parallelism_threads=12, inter_op_parallelism_threads=2,
                            allow_soft_placement=True)
    #
    sess = tf.Session(config=config)
    zfit.run.sess = sess
    # zfit.run.run_metadata = run_metadata
    # zfit.run.run_options = run_options
    # zfit.settings.set_verbosity(10)

    # testing = False
    testing = True
    # run_zfit = False
    run_zfit = True
    n_gauss_max = 9
    n_params_max = n_gauss_max
    # toys_nevents = [2 ** i for i in list(range(7, 18, 2)) + list(range(19, 24, 2))]
    toys_nevents = [2 ** i for i in list(range(7, 22, 2))]
    n_toys = 20

    if testing:
        n_gauss_max = 9
        # toys_nevents = [2**23]
        toys_nevents = [2097152]
        n_toys = 30
    results = {}
    results["n_toys"] = n_toys
    results["column"] = "number of gaussians"
    just_one = 0
    # for n_gauss in range(2, n_gauss_max + 1):
    # HACK START
    for n_gauss in [n_gauss_max]:
        # HACK END

        if n_gauss > n_gauss_max:
            break
        results[n_gauss] = {}
        results[n_gauss]["column"] = "number of free params"
        # for n_params in (1, n_gauss):
        # for n_params in (1,):
        for n_params in (n_gauss,):
            # HACK START
            # if just_one > 0:
            #     break
            # just_one += 1
            # HACK END
            if n_gauss < n_gauss_max and n_params not in (1, n_gauss):
                # HACK START
                pass
                # HACK END
                # continue  # only test the parameter scan for full params
            results_copy = results.copy()


            def intermediate_result_factory(res_tmp):
                results_copy[n_gauss][n_params] = res_tmp
                return results_copy


            # with tf.device("/device:GPU:0"):
            results[n_gauss][n_params] = toy_run(n_params=n_params, n_gauss=n_gauss,
                                                 n_toys=n_toys, toys_nevents=toys_nevents,
                                                 run_zfit=run_zfit,
                                                 intermediate_result_factory=intermediate_result_factory)

    # writer = tf.summary.FileWriter("tensorboard_log", graph=sess.graph)
    # writer.add_run_metadata(run_metadata, "my_session1")
    # writer.close()
    pprint.pprint(results)
    with open(f"{run_name}_{np.random.randint(low=0, high=int(1e1))}.yaml", "w") as f:
        yaml.dump(results, f)
