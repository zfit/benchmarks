"""Various code monitoring utilities."""

import os


def memory_usage():
    """Get memory usage of current process in MiB.

    Tries to use :mod:`psutil`, if possible, otherwise fallback to calling
    ``ps`` directly.

    Return:
        float: Memory usage of the current process.

    """
    pid = os.getpid()
    try:
        import psutil
        process = psutil.Process(pid)
        mem = process.memory_info()[0] / float(2 ** 20)
    except ImportError:
        import subprocess
        out = subprocess.Popen(['ps', 'v', '-p', str(pid)],
                               stdout=subprocess.PIPE).communicate()[0].split(b'\n')
        vsz_index = out[0].split().index(b'RSS')
        mem = float(out[1].split()[vsz_index]) / 1024
    return mem

# pylint: disable=too-few-public-methods

# EOF
