import argparse
import pprint
from collections import OrderedDict, defaultdict

import yaml
import numpy as np
import matplotlib.pyplot as plt


def process_results(file):
    with open(file) as result_file:
        result = yaml.load(result_file)
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
                avg_results[n_gauss][n_params][n_events] = (np.average(fit_result["success"]),
                                                            np.std(fit_result["success"]))

    return avg_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='evaluate gaussian toy results')
    # parser.add_argument('file', metavar='N', type=str, nargs='+',
    #                     help='an integer for the accumulator')
    # parser.add_argument('--sum', dest='accumulate', action='store_const',
    #                     const=sum, default=max,
    #                     help='sum the integers (default: find the max)')

    # args = parser.parse_args()
    # file = args.file[0]
    # file = "results/gauss_roofit/result_623883112794675899.yaml"
    file_grad = "results/gauss_zfit_grad/zfit_withgrad.yaml"
    file_nograd = "results/gauss_zfit_nograd/zfit_withgrad.yaml"
    # file = "zfitcpumkl_result_207406707159128210.yaml"
    # print(result)
    avg_results_nograd = process_results(file_nograd)
    avg_results_grad = process_results(file_grad)
    def difference(dict1, dict2):
        if isinstance(dict1, dict):
            diff = {}
            for key, value in dict1.items():
                try:
                    value2 = dict2[key]
                except KeyError:
                    continue
                else:
                    diff[key] = difference(value, value2)
        else:
            diff = (dict1[0] - dict2[0], np.sqrt(dict1[1]**2 + dict2[1]**2))
        return diff
    avg_results = difference(avg_results_nograd, avg_results_grad)
    pprint.pprint(avg_results)
    n_gausses_2param128 = []
    n_gausses_2param32768 = []
    n_gausses_2param2097152 = []

    n_gausses_nparam128 = []
    n_gausses_nparam32768 = []
    n_gausses_nparam2097152 = []

    n_gauss_2param_freeparam = []
    n_gauss_nparam_freeparam = []

    n_gausses_2param = defaultdict(list)
    n_gausses_2param_nevents = []

    n_gausses_nparam = defaultdict(list)
    n_gausses_nparam_nevents = []

    n_gausses = []

    plt.rc('axes', labelsize=18)  # fontsize of the x and y labels
    # plt.rcParams.update({'font.size': 16})

    for n_gauss, gauss_results in avg_results.items():
        if n_gauss == 20:
            continue
        n_gausses.append(n_gauss)
        for n_params, n_params_result in gauss_results.items():
            free_params = 2 * n_params
            if n_params == 1:
                n_gauss_2param_freeparam.append(free_params)
                for nevents, el in n_params_result.items():
                    n_gausses_2param[nevents].append(el[0])
            elif n_params == n_gauss:
                for nevents, el in n_params_result.items():
                    n_gausses_nparam[nevents].append(el[0])
            else:
                continue

            n_events = []
            times_err = []
            times = []
            for nevents, elapsed in n_params_result.items():
                n_events.append(nevents)
                times.append(elapsed[0])  # success only
                times_err.append(elapsed[1])


            plt.figure(f"figure_noscale_{n_params == 1}")
            # plt.loglog(n_events, times, label=f"n_gauss: {n_gauss}")
            # plt.plot(n_events, times, "x--", label=f"n_gauss: {n_gauss}")
            # plt.semilogx(n_events, times, label=f"n_gauss: {n_gauss}")
            # plt.loglog(n_events, times, "x--", label=f"n_gauss: {n_gauss}")
            ax = plt.axes()
            ax.set_xscale("log")
            # ax.set_yscale("log")
            plt.errorbar(n_events, times, yerr=times_err, fmt="x--", label=f"n_gauss: {n_gauss}")
            plt.legend()
            addition = f" and 2 free parameters" if free_params == 2 else ""
            plt.title(f"Toys with sum of gaussians" + addition)
            plt.xlabel("Number of events")
            plt.ylabel("Time (sec)")

    together = True

    if together:

        for nevents, times in n_gausses_2param.items():
            plt.figure("n_gauss_2param")
            # plt.semilogy(n_gausses, times, label=f"n events: {nevents}")
            # plt.plot(n_gausses, times, label=f"n events: {nevents}")
            plt.loglog(n_gausses, times, label=f"n events: {nevents}")
            plt.legend()
            # plt.title(f"Toys with sum of gaussians, total 2 free parameters.")
            plt.xlabel("Number of gaussians")
            plt.ylabel("Time (sec)")

        for nevents, times in n_gausses_nparam.items():
            continue
            plt.figure("n_gauss_nparam")
            n_gausses = np.array(n_gausses)
            n_params = 2 * n_gausses
            # plt.semilogy(n_params, times, label=f"n events: {nevents}")
            # plt.plot(n_params, times, label=f"n events: {nevents}")
            plt.loglog(n_params, times, label=f"n events: {nevents}")
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
