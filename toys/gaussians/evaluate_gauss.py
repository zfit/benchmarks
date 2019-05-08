import argparse

import yaml

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
    print(result)

    for n_gauss, gauss_results in result.items():
        pass