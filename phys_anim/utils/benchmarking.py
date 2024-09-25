# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import re
import sys
import time
from collections import defaultdict
from contextlib import contextmanager

average_times = defaultdict(lambda: (0, 0))


@contextmanager
def timeit(name):
    start = time.time()
    yield
    end = time.time()
    total_time, num_calls = average_times[name]
    total_time += end - start
    num_calls += 1
    print(
        "TIME:",
        name,
        end - start,
        "| AVG",
        total_time / num_calls,
        f"| TOTAL {total_time} {num_calls}",
    )
    average_times[name] = (total_time, num_calls)


def time_decorator(func):
    def with_times(*args, **kwargs):
        with timeit(func.__name__):
            return func(*args, **kwargs)

    return with_times


def recover_map(lines):
    info = {}
    pattern = re.compile(".* (.*) .* \| .* (.*\\b) .*\| .* (.*) (.*)")

    for l in lines:
        if not l.startswith("TIME"):
            continue

        match = pattern.match(l)

        name = match.group(1)
        avg = float(match.group(2))
        total_time = float(match.group(3))
        total_calls = float(match.group(4))
        info[name] = (avg, total_time, total_calls)

    return info


def compare_files(fileA, fileB):
    with open(fileA) as fA:
        linesA = fA.readlines()

    with open(fileB) as fB:
        linesB = fB.readlines()

    mapA = recover_map(linesA)
    mapB = recover_map(linesB)

    keysA = set(mapA.keys())
    keysB = set(mapB.keys())

    inter = keysA.intersection(keysB)
    print("Missing A", keysA.difference(inter))
    print("Missing B", keysB.difference(inter))

    keys_ordered = list(sorted([(mapA[k][1], k) for k in inter], reverse=True))

    for _, k in keys_ordered:
        print(f"{k} {mapA[k]} {mapB[k]}")


if __name__ == "__main__":
    fA = sys.argv[1]
    fB = sys.argv[2]
    compare_files(fA, fB)
