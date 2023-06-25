# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Facilities for reporting and collecting training statistics across
multiple processes and devices. The interface is designed to minimize
synchronization overhead as well as the amount of boilerplate in user
code."""

import re
import numpy as np
import torch
import dnnlib

from . import misc

#----------------------------------------------------------------------------

_num_moments    = 3             # [num_scalars, sum_of_scalars, sum_of_squares]
_reduce_dtype   = torch.float32 # Data type to use for initial per-tensor reduction.
_counter_dtype  = torch.float64 # Data type to use for the internal counters.
_rank           = 0             # Rank of the current process.
_sync_device    = None          # Device to use for multiprocess communication. None = single-process.
_sync_called    = False         # Has _sync() been called yet?
_counters       = dict()        # Running counters on each device, updated by report(): name => device => torch.Tensor
_cumulative     = dict()        # Cumulative counters on the CPU, updated by _sync(): name => torch.Tensor

#----------------------------------------------------------------------------

def init_multiprocessing(rank, sync_device):
    r"""Initializes `torch_utils.training_stats` for collecting statistics
    across multiple processes.

    This function must be called after
    `torch.distributed.init_process_group()` and before `Collector.update()`.
    The call is not necessary if multi-process collection is not needed.

    Args:
        rank:           Rank of the current process.
        sync_device:    PyTorch device to use for inter-process
                        communication, or None to disable multi-process
                        collection. Typically `torch.device('cuda', rank)`.
    """
    global _rank, _sync_device
    assert not _sync_called
    _rank = rank
    _sync_device = sync_device

#----------------------------------------------------------------------------

@misc.profiled_function
def report(name, value):
    r"""Broadcasts the given set of scalars to all interested instances of
    `Collector`, across device and process boundaries.

    This function is expected to be extremely cheap and can be safely
    called from anywhere in the training loop, loss function, or inside a
    `torch.nn.Module`.

    Warning: The current implementation expects the set of unique names to
    be consistent across processes. Please make sure that `report()` is
    called at least once for each unique name by each process, and in the
    same order. If a given process has no scalars to broadcast, it can do
    `report(name, [])` (empty list).

    Args:
        name:   Arbitrary string specifying the name of the statistic.
                Averages are accumulated separately for each unique name.
        value:  Arbitrary set of scalars. Can be a list, tuple,
                NumPy array, PyTorch tensor, or Python scalar.

    Returns:
        The same `value` that was passed in.
    """
    if name not in _counters:
        _counters[name] = dict()

    elems = torch.as_tensor(value)
    if elems.numel() == 0:
        return value

    elems = elems.detach().flatten().to(_reduce_dtype)
    moments = torch.stack([
        torch.ones_like(elems).sum(),
        elems.sum(),
        elems.square().sum(),
    ])
    assert moments.ndim == 1 and moments.shape[0] == _num_moments
    moments = moments.to(_counter_dtype)

    device = moments.device
    if device not in _counters[name]:
        _counters[name][device] = torch.zeros_like(moments)
    _counters[name][device].add_(moments)
    return value

#----------------------------------------------------------------------------

def report0(name, value):
    r"""Broadcasts the given set of scalars by the first process (`rank = 0`),
    but ignores any scalars provided by the other processes.
    See `report()` for further details.
    """
    report(name, value if _rank == 0 else [])
    return value

#----------------------------------------------------------------------------

class Collector:
    r"""Collects the scalars broadcasted by `report()` and `report0()` and
    computes their long-term averages (mean and standard deviation) over
    user-defined periods of time.

    The averages are first collected into internal counters that are not
    directly visible to the user. They are then copied to the user-visible
    state as a result of calling `update()` and can then be queried using
    `mean()`, `std()`, `as_dict()`, etc. Calling `update()` also resets the
    internal counters for the next round, so that the user-visible state
    effectively reflects averages collected between the last two calls to
    `update()`.

    Args:
        regex:          Regular expression defining which statistics to
                        collect. The default is to collect everything.
        keep_previous:  Whether to retain the previous averages if no
                        scalars were collected on a given round
                        (default: True).
    """
    def __init__(self, regex='.*', keep_previous=True):
        self._regex = re.compile(regex)
        self._keep_previous = keep_previous
        self._cumulative = dict()
        self._moments = dict()
        self.update()
        self._moments.clear()

    def names(self):
        r"""Returns the names of all statistics broadcasted so far that
        match the regular expression specified at construction time.
        """
        return [name for name in _counters if self._regex.fullmatch(name)]

    def update(self):
        r"""Copies current values of the internal counters to the
        user-visible state and resets them for the next round.

        If `keep_previous=True` was specified at construction time, the
        operation is skipped for statistics that have received no scalars
        since the last update, retaining their previous averages.

        This method performs a number of GPU-to-CPU transfers and one
        `torch.distributed.all_reduce()`. It is intended to be called
        periodically in the main training loop, typically once every
        N training steps.
        """
        if not self._keep_previous:
            self._moments.clear()
        for name, cumulative in _sync(self.names()):
            if name not in self._cumulative:
                self._cumulative[name] = torch.zeros([_num_moments], dtype=_counter_dtype)
            delta = cumulative - self._cumulative[name]
            self._cumulative[name].copy_(cumulative)
            if float(delta[0]) != 0:
                self._moments[name] = delta

    def _get_delta(self, name):
        r"""Returns the raw moments that were accumulated for the given
        statistic between the last two calls to `update()`, or zero if
        no scalars were collected.
        """
        assert self._regex.fullmatch(name)
        if name not in self._moments:
            self._moments[name] = torch.zeros([_num_moments], dtype=_counter_dtype)
        return self._moments[name]

    def num(self, name):
        r"""Returns the number of scalars that were accumulated for the given
        statistic between the last two calls to `update()`, or zero if
        no scalars were collected.
        """
        delta = self._get_delta(name)
        return int(delta[0])

    def mean(self, name):
        r"""Returns the mean of the scalars that were accumulated for the
        given statistic between the last two calls to `update()`, or NaN if
        no scalars were collected.
        """
        delta = self._get_delta(name)
        if int(delta[0]) == 0:
            return float('nan')
        return float(delta[1] / delta[0])

    def std(self, name):
        r"""Returns the standard deviation of the scalars that were
        accumulated for the given statistic between the last two calls to
        `update()`, or NaN if no scalars were collected.
        """
        delta = self._get_delta(name)
        if int(delta[0]) == 0 or not np.isfinite(float(delta[1])):
            return float('nan')
        if int(delta[0]) == 1:
            return float(0)
        mean = float(delta[1] / delta[0])
        raw_var = float(delta[2] / delta[0])
        return np.sqrt(max(raw_var - np.square(mean), 0))

    def as_dict(self):
        r"""Returns the averages accumulated between the last two calls to
        `update()` as an `dnnlib.EasyDict`. The contents are as follows:

            dnnlib.EasyDict(
                NAME = dnnlib.EasyDict(num=FLOAT, mean=FLOAT, std=FLOAT),
                ...
            )
        """
        stats = dnnlib.EasyDict()
        for name in self.names():
            stats[name] = dnnlib.EasyDict(num=self.num(name), mean=self.mean(name), std=self.std(name))
        return stats

    def __getitem__(self, name):
        r"""Convenience getter.
        `collector[name]` is a synonym for `collector.mean(name)`.
        """
        return self.mean(name)

#----------------------------------------------------------------------------

def _sync(names):
    r"""Synchronize the global cumulative counters across devices and
    processes. Called internally by `Collector.update()`.
    """
    if len(names) == 0:
        return []
    global _sync_called
    _sync_called = True

    # Collect deltas within current rank.
    deltas = []
    device = _sync_device if _sync_device is not None else torch.device('cpu')
    for name in names:
        delta = torch.zeros([_num_moments], dtype=_counter_dtype, device=device)
        for counter in _counters[name].values():
            delta.add_(counter.to(device))
            counter.copy_(torch.zeros_like(counter))
        deltas.append(delta)
    deltas = torch.stack(deltas)

    # Sum deltas across ranks.
    if _sync_device is not None:
        torch.distributed.all_reduce(deltas)

    # Update cumulative values.
    deltas = deltas.cpu()
    for idx, name in enumerate(names):
        if name not in _cumulative:
            _cumulative[name] = torch.zeros([_num_moments], dtype=_counter_dtype)
        _cumulative[name].add_(deltas[idx])

    # Return name-value pairs.
    return [(name, _cumulative[name]) for name in names]

#----------------------------------------------------------------------------
