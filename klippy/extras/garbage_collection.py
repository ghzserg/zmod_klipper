# Garbage collection optimizations
#
# Copyright (C) 2025  Branden Cash <ammmze@gmail.com>
# Copyright (C) 2016-2026  Kevin O'Connor <kevin@koconnor.net>
#
# This file may be distributed under the terms of the GNU GPLv3 license.
import gc
import logging
from types import FunctionType, MethodType

THRESHOLD = 0.05

# A generation sweep blocks the reactor
GC_SAFETY_FACTOR = 1.5
GC_TIME_MARGIN = 0.050
GC_INITIAL_COST = [0.005, 0.050, 0.500]

def get_function_owner(cb):
    if type(cb) in (FunctionType, MethodType):
        code = cb.__code__
        fn_name = getattr(cb, "__qualname__", None) or cb.__name__
        file = code.co_filename.rpartition("/")[2]
        line = code.co_firstlineno
        return "\'%s (%s:%d)\'" % (fn_name, file, line)
    return repr(cb)

def _analyze_callback(eventtime, prev_eventtime, cbs):
    duration = eventtime - prev_eventtime
    pretty_cbs = [get_function_owner(cb) for cb in cbs]
    logging.warning("Reactor %.3f busy for %.3f with:\n- %s" % (
        eventtime, duration, ',\n- '.join(pretty_cbs)))

class GarbageCollection:
    def __init__(self, config):
        self.printer = printer = config.get_printer()
        # Perform gc on reactor idle
        self._last_gc_times = [0., 0., 0.]
        self.defer_on_motion = config.getboolean('defer_gc_on_motion', False)
        self._gc_costs = list(GC_INITIAL_COST)
        self._skipped_gc = [0, 0, 0]
        self.motion_queuing = None
        self.reactor = reactor = printer.get_reactor()
        reactor.set_idle_notifier(self._handle_idle)
        printer.register_event_handler("klippy:analyze_shutdown",
                                       self._handle_analyze_shutdown)
        gc.disable()
        # feature check ... freeze/unfreeze is only available in python 3.7+
        can_freeze = hasattr(gc, 'freeze') and hasattr(gc, 'unfreeze')
        if can_freeze:
            printer.register_event_handler("klippy:ready", self._handle_ready)
            printer.register_event_handler("klippy:disconnect",
                                           self._handle_disconnect)
        printer.register_event_handler("klippy:ready",
                                       self._handle_ready_latency)

    def _handle_analyze_shutdown(self, msg, details):
        logging.info("Reactor garbage collection: %s (costs %s, deferred %s)",
                     self._last_gc_times, self._gc_costs, self._skipped_gc)

    def _get_gc_budget(self, eventtime):
        if self.motion_queuing is None:
            self.motion_queuing = self.printer.lookup_object('motion_queuing', None)
        if self.motion_queuing is None:
            return None
        return self.motion_queuing.get_step_gen_lead_time(eventtime)

    def _fits_in_budget(self, gc_level, budget):
        cost = self._gc_costs[gc_level] * GC_SAFETY_FACTOR + GC_TIME_MARGIN
        return cost <= budget

    def _handle_idle(self, eventtime, start_busy_time):
        gi = gc.get_count()
        if gi[0] < 700:
            return False
        # Reactor is idle and gc is due - run it
        gc_level = 0
        if gi[1] >= 10:
            gc_level = 1
            if gi[2] >= 10:
                gc_level = 2
        self._last_gc_times[gc_level] = eventtime
        gc.collect(gc_level)
        return True

    def _handle_ready(self):
        logging.debug("Running full garbage collection and freezing")
        for n in range(3):
            gc.collect(n)
        gc.freeze()

    def _handle_ready_latency(self):
        reactor = self.printer.get_reactor()
        def set_latency_notifier(eventtime):
            reactor.set_latency_notifier(THRESHOLD, _analyze_callback)
        reactor.register_callback(set_latency_notifier)

    def _handle_disconnect(self):
        logging.debug("Unfreezing garbage collection")
        gc.unfreeze()

def load_config(config):
    return GarbageCollection(config)
