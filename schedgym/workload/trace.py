"""trace - A trace-based workload generator

Inherits from the base WorkloadGenerator and uses the swf_parser to parse SWF
files.
"""

import copy
from collections.abc import Callable, Iterator, Sequence

from ..job import Job
from .base import WorkloadGenerator
from .swf_parser import parse as parse_swf


class TraceGenerator(WorkloadGenerator):
    restart: bool
    trace: Sequence[Job]
    refresh_jobs: Callable | None = None

    def __init__(
        self, restart: bool = False, trace: Sequence[Job] | None = None
    ) -> None:
        self.current_time = 0
        self.restart = restart
        self.current_element = 0
        # Untouched copy of the trace, taken on the first step (after any
        # rescaling done by the caller), used to start each new lap fresh.
        self._pristine: list[Job] | None = None

        if trace is not None:
            self.trace = trace
        else:
            self.trace = []

    def _restart_lap(self) -> None:
        """Start a new lap of the trace at the current time.

        Each lap is a fresh deep copy of the original jobs, shifted by the time
        the lap starts. Re-using the job objects of the previous lap would keep
        their run state, and adding the lap start to already shifted times
        would add a growing idle gap before every lap after the second.
        """
        assert self._pristine is not None
        base = self.current_time
        self.trace = copy.deepcopy(self._pristine)
        for job in self.trace:
            job.submission_time += base
        self.current_element = 0

    def step(self, offset: int = 1) -> list[Job | None]:
        """ "Samples" jobs from the trace file.

        Parameters
        ----------
            offset : int
                The amount to offset the current time step
        """
        if offset < 0:
            raise ValueError("Submission time must be positive")
        if self.restart and self.refresh_jobs is None and self._pristine is None:
            t0 = self.trace[0].submission_time if self.trace else 0
            self._pristine = copy.deepcopy(list(self.trace))
            for job in self._pristine:
                job.submission_time -= t0
        if self.current_element >= len(self.trace):
            if self.restart:
                if self.refresh_jobs is None:
                    self._restart_lap()
                else:
                    # generated workloads draw a new job list for each lap
                    self.current_element = 0
                    for job in self.trace:
                        job.submission_time += self.current_time
                    self.refresh_jobs()
            else:
                raise StopIteration("Workload finished")
        submission_time = self.current_time + offset
        self.current_time = submission_time
        if self.trace[self.current_element].submission_time > submission_time:
            return []

        start = self.current_element
        end = start
        trace = self.trace
        trace_len = len(trace)
        while end < trace_len and trace[end].submission_time <= submission_time:
            end += 1

        self.current_element = end
        return list(trace[start:end])

    @property
    def last_event_time(self) -> int:
        """The submission time of the last generated job"""
        offset = self.current_element if self.current_element < len(self.trace) else -1
        return self.trace[offset].submission_time

    def __len__(self) -> int:
        return len(self.trace)

    def __next__(self) -> Job:
        if self.current_element >= len(self.trace):
            if self.restart:
                self.current_element = 0
                if self.refresh_jobs is not None:
                    self.refresh_jobs()
            else:
                raise StopIteration()
        job = self.trace[self.current_element]
        self.current_element += 1
        return job

    def __iter__(self) -> Iterator[Job | None]:
        return iter(self.trace)

    def peek(self) -> Job | None:
        job = next(self)
        if self.current_element > 0:
            self.current_element -= 1
        return job


class SwfGenerator(TraceGenerator):
    """A trace-based (workload log) generator.

    Supports starting the parsing after an offset, and also supports reading a
    pre-specified number of jobs.

    Parameters
    ----------
        tracefile : str
            The path to the filed to be parsed and used as input for workload
            generation.
        processors : int
            The number of processors in this trace
        memory : int
            The amount of memory in this trace
        restart : bool
            Whether to restart from the beginning of the file when we reach
            its end (or, in the case we're using an offset and a length, to
            restart from the offset up to the length)
        ignore_memory : bool
            Whether to ignore (or not) memory usage
    """

    tracefile: str
    ignore_memory: bool

    def __init__(
        self,
        tracefile,
        processors,
        memory,
        offset=0,
        length=None,
        restart=False,
        ignore_memory=False,
    ):
        super().__init__(
            restart,
            list(parse_swf(tracefile, processors, memory, ignore_memory)),
        )
        self.tracefile = tracefile

        if length is None:
            length = len(self.trace)
        else:
            length = length if length <= len(self.trace) else len(self.trace)

        self.trace = self.trace[offset : offset + length]

        self.current_element = 0
