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

import time
from operator import itemgetter

from phys_anim.utils.common import print_info


class Timer:
    def __init__(self, name):
        self.name = name
        self.start_time = None
        self.time_total = 0.0
        self.num_ons = 0

    def on(self):
        assert self.start_time is None, "Timer {} is already turned on!".format(
            self.name
        )
        self.num_ons += 1
        self.start_time = time.time()

    def off(self):
        assert self.start_time is not None, "Timer {} not started yet!".format(
            self.name
        )
        self.time_total += time.time() - self.start_time
        self.start_time = None

    def report(self):
        if self.num_ons > 0:
            print_info(
                "Time report [{}]: {:.2f} {:.4f} seconds".format(
                    self.name, self.time_total, self.time_total / self.num_ons
                )
            )

    def clear(self):
        self.start_time = None
        self.time_total = 0.0


class TimeReport:
    def __init__(self):
        self.timers = {}

    def add_timer(self, name):
        assert name not in self.timers, "Timer {} already exists!".format(name)
        self.timers[name] = Timer(name=name)

    def start_timer(self, name):
        assert name in self.timers, "Timer {} does not exist!".format(name)
        self.timers[name].on()

    def end_timer(self, name):
        assert name in self.timers, "Timer {} does not exist!".format(name)
        self.timers[name].off()

    def report(self, name=None):
        if name is not None:
            assert name in self.timers, "Timer {} does not exist!".format(name)
            self.timers[name].report()
        else:
            print_info("------------Time Report------------")

            timer_with_times = []
            for timer_name in self.timers.keys():
                timer_with_times.append(
                    (self.timers[timer_name].time_total, self.timers[timer_name])
                )
            timer_with_times.sort(key=itemgetter(0))

            for _, timer in timer_with_times:
                timer.report()
            print_info("-----------------------------------")

    def clear_timer(self, name=None):
        if name is not None:
            assert name in self.timers, "Timer {} does not exist!".format(name)
            self.timers[name].clear()
        else:
            for timer_name in self.timers.keys():
                self.timers[timer_name].clear()

    def pop_timer(self, name=None):
        if name is not None:
            assert name in self.timers, "Timer {} does not exist!".format(name)
            self.timers[name].report()
            del self.timers[name]
        else:
            self.report()
            self.timers = {}
