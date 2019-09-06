import pprint

import progressbar
import yaml
import zfit
import zfit.minimizers.baseminimizer
import numpy as np

import zfit_benchmark

zfit.run.numeric_checks = False


def toy_run(n_params, n_gauss, nevents):
    # pdf = chebys[0]

    # zfit.settings.set_verbosity(10)

    lower = -1
    upper = 1
    # create observables
    obs = zfit.Space("obs1", limits=(lower, upper))

    # create parameters
    params = []
    params_initial = []
    mu_lower, mu_upper = 1, 3
    sigma_lower, sigma_upper = 0.5, 2
    for i in range(n_params):
        mu = zfit.Parameter(f"mu_{i}_{nevents}", np.random.uniform(low=mu_lower, high=mu_upper), mu_lower,
                            mu_upper)
        sigma = zfit.Parameter(f"sigma_{i}_{nevents}", np.random.uniform(low=sigma_lower, high=sigma_upper),
                               sigma_lower, sigma_upper)
        params.append((mu, sigma))
    # create pdfs
    pdfs = []
    for i in range(n_gauss):
        mu, sigma = params[i % n_params]
        shifted_mu = mu + 0.3 * i
        shifted_sigma = sigma + 0.1 * i
        pdf = zfit.pdf.Gauss(obs=obs, mu=shifted_mu, sigma=shifted_sigma)
        # from zfit.models.basic import CustomGaussOLD
        # pdf = CustomGaussOLD(obs=obs, mu=shifted_mu, sigma=shifted_sigma)
        # pdf.update_integration_options(mc_sampler=tf.random_uniform)
        pdfs.append(pdf)
    initial_param_val = 1 / n_gauss
    fracs = []
    for i in range(n_gauss - 1):
        frac_value = 1 / n_gauss
        lower_value = 0.0001
        upper_value = 1.5 / n_gauss
        frac = zfit.Parameter(f"frac_{i}", value=1 / n_gauss, lower_limit=lower_value, upper_limit=upper_value)
        frac.floating = False
        fracs.append(frac)
    sum_pdf = zfit.pdf.SumPDF(pdfs=pdfs, fracs=fracs)
    # sum_pdf.update_integration_options(mc_sampler=tf.random_uniform)
    pdf = sum_pdf

    # Create dictionary to save fit results
    failed_fits = 0
    successful_fits = 0

    sampler = pdf.create_sampler(n=nevents, fixed_params=True)
    sampler.set_data_range(obs)
    nll = zfit.loss.UnbinnedNLL(pdf, sampler)

    minimizer = zfit.minimize.MinuitMinimizer(zfit.minimizers.baseminimizer.ToyStrategyFail(), verbosity=0)
    minimizer._use_tfgrad = True

    timer = zfit_benchmark.timer.Timer(f"Timing")

    sampler.resample()
    with timer:
        to_run = [nll.value(), nll.gradients()]
        zfit.run(to_run)
    return
    dependents = pdf.get_dependents()

    # HACK stop here
    with timer:
        with timer.child(f"toy gauss gpu") as child:
            sampler.resample()
            for param in dependents:
                param.randomize()
            # with tf.device("/device:GPU:0"):
            minimum = minimizer.minimize(nll)

        if minimum.converged:
            successful_fits += 1
            fail_or_success = "success"
        else:
            failed_fits += 1
            fail_or_success = "fail"


if __name__ == '__main__':
    import tensorflow as tf

    sess = tf.Session()
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    # # sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    # zfit.run.sess = sess
    # zfit.run.run_metadata = run_metadata
    # zfit.run.run_options = run_options


    # random_uniform = tf.random_uniform(shape=(199,))
    # from tensorflow.python.client import timeline
    # rnd = tf.sqrt(random_uniform)
    # rnd = tf.log(tf.abs(rnd))
    # with tf.Session() as sess:
    #     options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    #     run_metadata = tf.RunMetadata()
    #     sess.run(rnd, options=options, run_metadata=run_metadata)
    #
    #     # Create the Timeline object, and write it to a json file
    #     fetched_timeline = timeline.Timeline(run_metadata.step_stats)
    #     chrome_trace = fetched_timeline.generate_chrome_trace_format()
    #     with open('/home/jonas/tmp/timeline_01.json', 'w') as f:
    #         f.write(chrome_trace)
    #     writer = tf.summary.FileWriter("tensorboard_log", graph=sess.graph)
    #
    #     writer.add_run_metadata(run_metadata, "my_session1")
    #     writer.close()
    # zfit.run(rnd)

    n_gauss = 3
    n_params = 3
    n_events = 500000

    # with tf.device("/device:GPU:0"):


    # pdf = chebys[0]

    # zfit.settings.set_verbosity(10)

    lower = -1
    upper = 1
    # create observables
    obs = zfit.Space("obs1", limits=(lower, upper))

    # create parameters
    params = []
    params_initial = []
    mu_lower, mu_upper = 1, 3
    sigma_lower, sigma_upper = 0.5, 2
    for i in range(n_params):
        mu = zfit.Parameter(f"mu_{i}_{n_events}", np.random.uniform(low=mu_lower, high=mu_upper), mu_lower,
                            mu_upper)
        sigma = zfit.Parameter(f"sigma_{i}_{n_events}", np.random.uniform(low=sigma_lower, high=sigma_upper),
                               sigma_lower, sigma_upper)
        params.append((mu, sigma))
    # create pdfs
    pdfs = []
    for i in range(n_gauss):
        mu, sigma = params[i % n_params]
        shifted_mu = mu + 0.3 * i
        shifted_sigma = sigma + 0.1 * i
        pdf = zfit.pdf.Gauss(obs=obs, mu=shifted_mu, sigma=shifted_sigma)
        # from zfit.models.basic import CustomGaussOLD
        # pdf = CustomGaussOLD(obs=obs, mu=shifted_mu, sigma=shifted_sigma)
        # pdf.update_integration_options(mc_sampler=tf.random_uniform)
        pdfs.append(pdf)
    initial_param_val = 1 / n_gauss
    fracs = []
    for i in range(n_gauss - 1):
        frac_value = 1 / n_gauss
        lower_value = 0.0001
        upper_value = 1.5 / n_gauss
        frac = zfit.Parameter(f"frac_{i}", value=1 / n_gauss, lower_limit=lower_value, upper_limit=upper_value)
        frac.floating = False
        fracs.append(frac)
    sum_pdf = zfit.pdf.SumPDF(pdfs=pdfs, fracs=fracs)
    # sum_pdf.update_integration_options(mc_sampler=tf.random_uniform)
    pdf = sum_pdf

    # Create dictionary to save fit results
    failed_fits = 0
    successful_fits = 0

    sampler = pdf.create_sampler(n=n_events, fixed_params=True)
    sampler.set_data_range(obs)
    nll = zfit.loss.UnbinnedNLL(pdf, sampler)

    minimizer = zfit.minimize.MinuitMinimizer(zfit.minimizers.baseminimizer.ToyStrategyFail(), verbosity=0)
    minimizer._use_tfgrad = True

    timer = zfit_benchmark.timer.Timer(f"Timing")

    sampler.resample()
    # to_run = [nll.value(), nll.gradients()]
    to_run = [nll.value()]
    zfit.run(to_run)

    # zfit.run(to_run)
    from tensorflow.python.client import timeline
    # with tf.Session() as sess:
    # sess.run(tf.global_variables_initializer())
    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    # sess.run(to_run, options=options, run_metadata=run_metadata)
    with timer:
        for _ in range(1):

            # val = zfit.run(to_run, options=options, run_metadata=run_metadata)
            val = zfit.run(sampler.sample_holder.initializer, options=options, run_metadata=run_metadata)
            # val = zfit.run(to_run)
    print(f"Time needed for single run: {timer.elapsed}")

    # Create the Timeline object, and write it to a json file
    fetched_timeline = timeline.Timeline(run_metadata.step_stats)
    chrome_trace = fetched_timeline.generate_chrome_trace_format()
    with open('/home/jonas/tmp/timeline_01.json', 'w') as f:
        f.write(chrome_trace)
    writer = tf.summary.FileWriter("tensorboard_log", graph=sess.graph)

    writer.add_run_metadata(run_metadata, "my_session1")
    writer.close()


    # writer.add_run_metadata(run_metadata, "my_session1")
    # writer.close()
