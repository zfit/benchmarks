import argparse
import pprint

import yaml
import numpy as np
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
    avg_results = {}
    n_toys = result.pop("n_toys", None)
    column_n_gauss = result.pop("column", None)
    for n_gauss, gauss_results in result.items():
        avg_results[n_gauss] = {}
        column_n_free_params = gauss_results.pop("column", None)
        for n_params, params_results in gauss_results.items():
            avg_results[n_gauss][n_params] = {}
            column_n_events = params_results.pop("column", None)
            for n_events, fit_result in params_results.items():
                avg_results[n_gauss][n_params][n_events] = (np.average(fit_result["success"]), np.std(fit_result["success"]))

    pprint.pprint(avg_results)
