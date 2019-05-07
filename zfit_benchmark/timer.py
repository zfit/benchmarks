import math
from collections import OrderedDict
from timeit import default_timer

# class Timer(object):
#     """Time the code placed inside its context.
#
#     Taken from http://coreygoldberg.blogspot.ch/2012/06/python-timer-class-context-manager-for.html
#
#     Attributes:
#         verbose (bool): Print the elapsed time at context exit?
#         start (float): Start time in seconds since Epoch Time. Value set
#             to 0 if not run.
#         elapsed (float): Elapsed seconds in the timer. Value set to
#             0 if not run.
#
#     Arguments:
#         verbose (bool, optional): Print the elapsed time at
#             context exit? Defaults to False.
#
#     """
#
#     def __init__(self, verbose=False):
#         """Initialize the timer."""
#         self.verbose = verbose
#         self._timer = default_timer
#         self.start = 0
#         self.elapsed = 0
#
#     def __enter__(self):
#         self.start = self._timer()
#         return self
#
#     def __exit__(self, *args):
#         self.elapsed = self._timer() - self.start
#         if self.verbose:
#             print('Elapsed time: {} ms'.format(self.elapsed*1000.0))


from decimal import Decimal
from timeit import default_timer


# The code below is taken from https://github.com/mherrmann/timer-cm/blob/master/timer_cm.py
# and licensed with the MIT from mherrmann

# The following license applies for the code below
#
# MIT License
#
# Copyright (c) 2017 Michael Herrmann
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


class Timer:
    def __init__(self, name: str, do_print: bool = True):
        self.elapsed = Decimal()
        self._name = name
        self._do_print = do_print
        self._start_time = None
        self._children = OrderedDict()
        self._count = 0

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()
        if self._do_print:
            self.print_results()

    def child(self, name):
        try:
            return self._children[name]
        except KeyError:
            result = Timer(name, do_print=False)
            self._children[name] = result
            return result

    def start(self):
        self._count += 1
        self._start_time = self._get_time()

    def stop(self):
        self.elapsed += self._get_time() - self._start_time

    @property
    def elapsed(self):
        return self._elapsed + self._get_time() - self._start_time

    @elapsed.setter
    def elapsed(self, value):
        self._elapsed = value

    def print_results(self):
        print(self._format_results())

    def _format_results(self, indent='  '):
        children = self._children.values()
        elapsed = self.elapsed or sum(c.elapsed for c in children)
        result = f'{self._name}: {elapsed:.3f}s'
        max_count = max(c._count for c in children) if children else 0
        count_digits = 0 if max_count <= 1 else int(math.ceil(math.log10(max_count + 1)))
        for child in sorted(children, key=lambda c: c.elapsed, reverse=True):
            lines = child._format_results(indent).split('\n')
            child_percent = child.elapsed / elapsed * 100
            lines[0] += f' ({child_percent:.3f})'
            if count_digits:
                # `+2` for the 'x' and the space ' ' after it:
                lines[0] = (f'{child._count:}x ').rjust(count_digits + 2) \
                           + lines[0]
            for line in lines:
                result += '\n' + indent + line
        return result

    def _get_time(self):
        return Decimal(default_timer())
