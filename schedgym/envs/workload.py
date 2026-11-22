# flake8: noqa E501

import math
import random
import warnings
import itertools
from math import log2
from collections import namedtuple

from schedgym import workload as wl, job

JobParameters = namedtuple("JobParameters", ["small", "large"])


class DeepRmWorkloadGenerator(wl.DistributionalWorkloadGenerator):
    def __init__(self, *args: wl.BinomialWorkloadGenerator):
        super().__init__(max([w.length for w in args]))

        self.generators = args
        self.counter = itertools.count(1)

        for generator in self.generators:
            generator.counter = self.counter

    def step(self, offset=1) -> list[job.Job | None]:
        return self.generators[random.randint(0, len(self.generators) - 1)].step()

    def __len__(self):
        return self.generators[0].length

    def peek(self):
        return self.step()

    @staticmethod
    def build(
        new_job_rate,
        small_job_chance,
        max_job_len,
        max_job_size,
        ignore_memory=False,
        min_large_job_len=None,
        max_small_job_len=None,
        min_small_job_len=None,
        min_dominant_job_size=None,
        min_other_job_size=None,
        max_other_job_size=None,
        runtime_estimates=None,
        estimate_parameters=None,
    ) -> "DeepRmWorkloadGenerator":
        small_job_time_lower = 1 if min_small_job_len is None else min_small_job_len
        small_job_time_upper = (
            max(max_job_len // 5, 1) if max_small_job_len is None else max_small_job_len
        )
        large_job_time_lower = (
            int(max_job_len * (2 / 3))
            if min_large_job_len is None
            else min_large_job_len
        )
        large_job_time_upper = max_job_len

        dominant_resource_lower = (
            max_job_size // 2
            if min_dominant_job_size is None
            else min_dominant_job_size
        )
        dominant_resource_upper = max_job_size
        other_resource_lower = 1 if min_other_job_size is None else min_other_job_size
        other_resource_upper = (
            max_job_size // 5 if max_other_job_size is None else max_other_job_size
        )

        cpu_dominant_parameters = JobParameters(
            job.JobParameters(
                small_job_time_lower,
                small_job_time_upper,
                dominant_resource_lower,
                dominant_resource_upper,
                other_resource_lower,
                other_resource_upper,
            ),
            job.JobParameters(
                large_job_time_lower,
                large_job_time_upper,
                dominant_resource_lower,
                dominant_resource_upper,
                other_resource_lower,
                other_resource_upper,
            ),
        )

        mem_dominant_parameters = JobParameters(
            job.JobParameters(
                small_job_time_lower,
                small_job_time_upper,
                other_resource_lower,
                other_resource_upper,
                dominant_resource_lower,
                dominant_resource_upper,
            ),
            job.JobParameters(
                large_job_time_lower,
                large_job_time_upper,
                other_resource_lower,
                other_resource_upper,
                dominant_resource_lower,
                dominant_resource_upper,
            ),
        )

        generators = (
            wl.BinomialWorkloadGenerator(
                new_job_rate,
                small_job_chance,
                cpu_dominant_parameters.small,
                cpu_dominant_parameters.large,
                runtime_estimates=runtime_estimates,
                estimate_parameters=estimate_parameters,
            ),
            wl.BinomialWorkloadGenerator(
                new_job_rate,
                small_job_chance,
                mem_dominant_parameters.small,
                mem_dominant_parameters.large,
                runtime_estimates=runtime_estimates,
                estimate_parameters=estimate_parameters,
            ),
        )

        return DeepRmWorkloadGenerator(*generators[: (1 if ignore_memory else None)])


class SyntheticWorkloadGenerator(wl.TraceGenerator):
    """A synthetic workload generator based on realistic models."""

    def __init__(
        self,
        length,
        nodes,
        start_time=8,
        random_seed=0,
        restart=False,
        uniform_proportion=0.95,
        cdf_break=0.5,
        runtime_estimates=None,
        estimate_parameters=None,
    ):
        """Synthetic workload generator based on Lublin's work.

        Parameters
        ----------
            length : int
                number of jobs to generate
            nodes : int
                number of compute nodes in the system
            start_time : int
                hour of day in which to start simulation
            random_seed : int
                random seed to use to generate jobs
            restart : bool
                whether to restart after a sample finishes
            uniform_proportion : float
                tunes the proportion between the first and second uniform
                distributions in the two-stage uniform process
            cdf_break : float
                whether to move the break closer to the inferior or superior
                limit. A value closer to 0 will (tend to) produce bigger jobs,
                while a value closer to 1 will (tend to) produce smaller jobs
            runtime_estimates : {'gaussian', 'tsafrir', None}
                whether to include runtime estimates and the method used
                to compute them:
                * None generates perfect estimate (estimates equal run time)
                * 'gaussian' generates estimates with zero-mean Gaussian noise
                  added to them
                * 'tsafrir' uses Dan Tsafrir's model of user runtime estimates
                  to generate estimates
            estimate_parameters : Union[float, List[Tuple[float, float]]
                the parameters used for generating user estimates.
                Depends on :param:`runtime_estimates`.
                When `runtime_estimates` is 'gaussian', this is a single
                floating-point number that sets the standard deviation of the
                noise.
                When `runtime_estimates` is 'tsafrir', this is a list of
                floating-point pairs specifying a histogram (time, number of
                jobs) of job runtime popularity.
        """
        random.seed(random_seed)

        # Imported here so the DeepRM workload works without the compiled
        # parallelworkloads extension (it does not build on native Windows).
        from parallelworkloads.lublin99 import Lublin99

        self.lublin = Lublin99(False, random_seed, length)
        self.lublin.start = start_time
        self.random_seed = random_seed
        self.nodes = nodes

        uniform_low_prob = 0.8
        log2_size = log2(nodes)
        min_umed = log2_size - 3.5
        max_umed = log2_size - 1.5
        breaking_point = cdf_break * min_umed + (1 - cdf_break) * max_umed

        self.lublin.setParallelJobProbabilities(
            False,
            uniform_low_prob,
            breaking_point,
            log2_size,
            uniform_proportion,
        )

        self.runtime_estimates = runtime_estimates
        self.estimate_parameters = estimate_parameters

        trace = self.refresh_jobs()
        super().__init__(restart, trace)

    def refresh_jobs(self):
        """Refreshes the underlying job list."""
        jobs = self.lublin.generate()
        if self.runtime_estimates:
            if self.runtime_estimates == "tsafrir":
                if self.estimate_parameters is not None:
                    warnings.warn("Setting tsafrir parameters is currently unsupported")
                from parallelworkloads.tsafrir05 import Tsafrir05

                tsafrir = Tsafrir05(jobs)
                jobs = tsafrir.generate(jobs)
            elif self.runtime_estimates == "gaussian":
                for j in jobs:
                    j.reqTime = math.ceil(
                        random.gauss(j.runTime, self.estimate_parameters * j.runTime)
                    )
                    if j.reqTime < 1:
                        j.reqTime = 1
            else:
                raise ValueError(f"Unsupported estimate type {self.runtime_estimates}")

        self.trace = [job.Job.from_swf_job(j) for j in jobs]
        return self.trace


def build_swf(
    tracefile,
    processors=None,
    cluster_processors=None,
    time_scale=1,
    offset=0,
    length=None,
    restart=True,
    ignore_memory=True,
):
    """Trace workload from a Standard Workload Format (SWF) file (.swf or .swf.gz).

    Parameters
    ----------
        tracefile : str
            Path to the SWF log, e.g. one from the Parallel Workloads Archive.
        processors : int, optional
            Processors of the traced machine; read from the ``; MaxProcs:``
            header when omitted.
        cluster_processors : int, optional
            Processors of the simulated cluster (the env's ``processors``).
            When given, job sizes are rescaled proportionally
            (``ceil(size * cluster_processors / processors)``) so that the trace
            keeps its relative load on a smaller cluster.
        time_scale : int
            Seconds per simulated time unit. SWF times are in seconds; the
            simulator advances one unit per step, so e.g. ``60`` turns the trace
            into minutes. Times are rounded up and floored at 1.
        offset, length : int
            Use jobs ``[offset, offset + length)`` of the trace.
        restart : bool
            Loop over the trace when it ends (needed for continuing training).
        ignore_memory : bool
            Most SWF logs have no usable memory field; keep True and run the
            env with ``ignore_memory=True``.
    """
    from schedgym.workload import SwfGenerator

    gen = SwfGenerator(
        tracefile,
        processors,
        0,
        offset=offset,
        length=length,
        restart=restart,
        ignore_memory=ignore_memory,
    )
    trace_procs = processors or max(j.requested_processors for j in gen.trace)
    t0 = gen.trace[0].submission_time if gen.trace else 0
    for j in gen.trace:
        j.submission_time = (j.submission_time - t0) // time_scale
        j.execution_time = max(1, math.ceil(j.execution_time / time_scale))
        j.requested_time = max(
            j.execution_time, math.ceil(j.requested_time / time_scale)
        )
        if cluster_processors:
            j.requested_processors = min(
                cluster_processors,
                max(1, math.ceil(j.requested_processors * cluster_processors / trace_procs)),
            )
            j.processors_allocated = j.requested_processors
        if ignore_memory:
            j.requested_memory = 0
    return gen


def build(workload_config: dict):
    type = workload_config["type"]
    kwargs = {k: v for k, v in workload_config.items() if k != "type"}
    if type == "deeprm":
        return DeepRmWorkloadGenerator.build(**kwargs)
    elif type == "lublin":
        return SyntheticWorkloadGenerator(**kwargs)
    elif type == "swf":
        return build_swf(**kwargs)
    else:
        raise RuntimeError(f"Unsupported workload model type {type} requested")
