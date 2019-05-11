import argparse
import pprint
from collections import OrderedDict

import yaml
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='evaluate gaussian toy results')
    parser.add_argument('file', metavar='N', type=str, nargs='+',
                        help='an integer for the accumulator')
    # parser.add_argument('--sum', dest='accumulate', action='store_const',
    #                     const=sum, default=max,
    #                     help='sum the integers (default: find the max)')

    args = parser.parse_args()
    with open(args.file[0]) as result_file:
        result = yaml.load(result_file)
    # print(result)
    avg_results = OrderedDict()
    finished = result.pop("ATTENTION", False) == "ATTENTION"
    n_toys = result.pop("n_toys", None)
    column_n_gauss = result.pop("column", None)
    for n_gauss, gauss_results in result.items():
        avg_results[n_gauss] = OrderedDict()
        column_n_free_params = gauss_results.pop("column", None)
        for n_params, params_results in gauss_results.items():
            avg_results[n_gauss][n_params] = OrderedDict()
            column_n_events = params_results.pop("column", None)
            for n_events, fit_result in params_results.items():
                avg_results[n_gauss][n_params][n_events] = (
                np.average(fit_result["success"]), np.std(fit_result["success"]))

    pprint.pprint(avg_results)
    n_gausses_2param128 = []
    n_gausses_2param32768 = []
    n_gausses_2param2097152 = []

    n_gausses_nparam128 = []
    n_gausses_nparam32768 = []
    n_gausses_nparam2097152 = []

    n_gauss_2param_freeparam = []
    n_gauss_nparam_freeparam = []

    n_gausses_2param = []
    n_gausses_2param_nevents = []

    n_gausses_nparam = []
    n_gausses_nparam_nevents = []

    n_gausses = []

    for n_gauss, gauss_results in avg_results.items():
        if n_gauss == 20:
            continue
        n_gausses.append(n_gauss)
        for n_params, n_params_result in gauss_results.items():
            free_params = 2 * n_params
            if n_params == 1:
                n_gauss_2param_freeparam.append(free_params)
                n_gausses_2param.append([el[0] for el in n_params_result.values()])
                n_gausses_2param_nevents = list(n_params_result.keys())
                n_gausses_2param128.append(n_params_result[128][0])
                n_gausses_2param32768.append(n_params_result[32768][0])
                n_gausses_2param2097152.append(n_params_result[2097152][0])

            if n_params == n_gauss:
                n_gausses_nparam.append([el[0] for el in n_params_result.values()])
                n_gausses_nparam_nevents = list(n_params_result.keys())
                n_gauss_nparam_freeparam.append(free_params)
                n_gausses_nparam128.append(n_params_result[128][0])
                n_gausses_nparam32768.append(n_params_result[32768][0])
                n_gausses_nparam2097152.append(n_params_result[2097152][0])

            n_events = []
            times = []
            for nevents, elapsed in n_params_result.items():
                n_events.append(nevents)
                times.append(elapsed[0])  # success only

            plt.figure(f"figure_noscale_{n_params == 1}")
            plt.loglog(n_events, times, label=f"n_gauss: {n_gauss}")
            plt.legend()
            addition = f"and 2 free parameters" if free_params == 2 else ""
            plt.title(f"Toys with sum of gaussians" + addition)
            plt.xlabel("Number of events")
            plt.ylabel("Time (sec)")

    together = True

    if together:

        for times, nevents in zip(n_gausses_2param, n_gausses_2param_nevents):
            plt.figure("n_gauss_2param")
            plt.semilogy(n_gausses, times, label=f"n events: {nevents}")
            plt.legend()
            plt.title(f"Toys with sum of gaussians, total 2 free parameters.")
            plt.xlabel("Number of gaussians")
            plt.ylabel("Time (sec)")

        for times, nevents in zip(n_gausses_nparam, n_gausses_nparam_nevents):
            plt.figure("n_gauss_nparam")
            n_gausses = np.array(n_gausses)
            n_params = 3 * n_gausses - 1
            plt.semilogy(n_params, times, label=f"n events: {nevents}")
            plt.legend()
            plt.title(f"Toys with sum of gaussians")
            plt.xlabel("Number of free params")
            plt.ylabel("Time (sec)")
    else:
        for times, nevents in (
        (n_gausses_2param128, 128), (n_gausses_2param32768, 32768), (n_gausses_2param2097152, 2097152)):
            plt.figure()
            plt.plot(n_gausses, times)
            plt.title(f"Toys with {nevents} and sum of gaussians with 2 free parameters.")
            plt.xlabel("Number of gaussians")
            plt.ylabel("Time (sec)")

        for times, nevents in (
        (n_gausses_nparam128, 128), (n_gausses_nparam32768, 32768), (n_gausses_nparam2097152, 2097152)):
            plt.figure()
            n_gausses = np.array(n_gausses)
            n_params = 3 * n_gausses - 1
            plt.plot(n_params, times, "x")
            plt.title(f"Toys with {nevents} and sum of gaussians")
            plt.xlabel("Number of free params")
            plt.ylabel("Time (sec)")

    plt.show()
