from collections.abc import Callable
from enum import IntEnum
from typing import cast

from schedgym.envs.workload import (
    DeepRmWorkloadGenerator,
    SyntheticWorkloadGenerator,
)
from schedgym.job import Job
from schedgym.scheduler import NullScheduler
from schedgym.workload import TraceGenerator

WorkloadGeneratorType = DeepRmWorkloadGenerator | TraceGenerator


class SimulationType(IntEnum):
    EVENT_BASED = 0
    TIME_BASED = 1

    @staticmethod
    def from_str(simulation_type: str):
        simulation_type = simulation_type.upper().replace("-", "_")
        if simulation_type in SimulationType.__members__:
            return SimulationType[simulation_type]
        else:
            raise ValueError(f"{simulation_type} is not a valid SimulationType.")


WAIT_MODES = ("unit", "next_event")


class DeepRmSimulator:
    scheduler: NullScheduler
    workload: DeepRmWorkloadGenerator | SyntheticWorkloadGenerator

    def __init__(
        self,
        workload_generator: WorkloadGeneratorType,
        scheduler: NullScheduler,
        simulation_type: SimulationType = SimulationType.TIME_BASED,
        job_slots: int | None = None,
        wait_mode: str = "unit",
    ):
        if wait_mode not in WAIT_MODES:
            raise ValueError(f"wait_mode must be one of {WAIT_MODES}, not {wait_mode!r}")
        self.wait_mode = wait_mode
        self.scheduler = scheduler
        self.workload = workload_generator
        self.simulation_type = simulation_type
        self.job_slots = slice(0, job_slots)
        self.simulator = self.build()
        self.reset(self.workload, scheduler)

    def rl_step(
        self,
        action: int | None,
        listjobs: Callable | None,
    ) -> list[list[Job]]:
        return self.simulator.rl_step(
            action if action is not None else -1,
            listjobs if listjobs else lambda: self.scheduler.jobs_in_system,
        )

    def build(self):
        if self.simulation_type == SimulationType.EVENT_BASED:
            return EventBasedDeepRmSimulator(
                self.workload,
                self.scheduler,
                self.job_slots,
                wait_mode=self.wait_mode,
            )
        elif self.simulation_type == SimulationType.TIME_BASED:
            return TimeBasedDeepRmSimulator(
                self.workload,
                self.scheduler,
                self.job_slots,
            )
        else:
            raise NotImplementedError(
                f"Unsupported simulation type {self.simulation_type}"
            )

    @property
    def current_time(self):
        return self.simulator.current_time

    @property
    def last_job_time(self):
        return self.simulator.last_job_time

    def reset(self, workload, scheduler):
        self.scheduler = scheduler
        self.workload = workload
        self.simulator = self.build()


class WorkloadExhausted(StopIteration):
    """The workload ran out in the middle of a transition.

    ``ticks`` holds the jobs listed at every time unit that did elapse before
    the workload ran out, so the environment can still charge for them.
    """

    def __init__(self, ticks: list[list[Job]]):
        super().__init__("Workload finished")
        self.ticks = ticks


class EventBasedDeepRmSimulator:
    """Event-based (SMDP) simulator: the agent is asked only when a job fits.

    ``wait_mode`` says how long the "wait" action lasts when a job already
    fits: ``"unit"`` advances one time unit, ``"next_event"`` advances until
    a job arrives or finishes.
    """

    last_job_time: int
    scheduler: NullScheduler
    job_slots: slice

    def __init__(
        self,
        workload_generator: WorkloadGeneratorType,
        scheduler: NullScheduler,
        job_slots: slice,
        wait_mode: str = "unit",
    ):
        if (
            not isinstance(workload_generator, DeepRmWorkloadGenerator)
            and not isinstance(workload_generator, TraceGenerator)
        ) or not isinstance(scheduler, NullScheduler):
            raise AssertionError("Invalid arguments received.")

        self.current_time = 0
        self.scheduler = scheduler
        self.simulation_start_time = 0
        self.workload = workload_generator
        self.job_slots = job_slots
        self.wait_mode = wait_mode

        self.current_time = self.last_job_time = 0
        if isinstance(workload_generator, TraceGenerator):
            first_job_time = cast(Job, workload_generator.peek()).submission_time - 1
            workload_generator.current_time = first_job_time
            scheduler.job_events.time = first_job_time
            scheduler.current_time = first_job_time
            self.current_time = first_job_time

    def rl_step(
        self, action: int, listjobs: Callable[[], list[Job]]
    ) -> list[list[Job]]:
        "Returns a list of jobs for each successful intermediate time step."

        if self.scheduler.step(action):
            return [[]]

        jobs: list[list[Job]] = []
        completed_before = len(self.scheduler.queue_completed)
        event = self.wait_mode == "unit"  # "unit": any time unit is enough
        self.current_time += 1
        while True:
            try:
                j = self.workload.step()
            except StopIteration as e:
                raise WorkloadExhausted(jobs) from e
            if j:
                self.scheduler.submit(j)
                self.last_job_time = self.current_time
                event = True
            self.scheduler.forward_time()
            jobs.append(listjobs())
            if len(self.scheduler.queue_completed) > completed_before:
                event = True
            if event and self.scheduler.some_job_fits(self.job_slots):
                break
        return jobs


class TimeBasedDeepRmSimulator:
    last_job_time: int
    scheduler: NullScheduler
    job_slots: slice

    def __init__(
        self,
        workload_generator: WorkloadGeneratorType,
        scheduler: NullScheduler,
        job_slots: slice,
    ):
        if (
            not isinstance(workload_generator, DeepRmWorkloadGenerator)
            and not isinstance(workload_generator, TraceGenerator)
        ) or not isinstance(scheduler, NullScheduler):
            raise AssertionError("Invalid arguments received.")

        self.scheduler = scheduler
        self.simulation_start_time = 0
        self.workload = workload_generator
        self.current_time = self.last_job_time = 0
        self.job_slots = job_slots

        if isinstance(workload_generator, TraceGenerator):
            first_job_time = cast(Job, workload_generator.peek()).submission_time - 1
            workload_generator.current_time = first_job_time
            scheduler.job_events.time = first_job_time
            scheduler.current_time = first_job_time

    def step(self, _=True):
        """Not implemented in DeepRmSimulator"""
        raise NotImplementedError("This simulator cannot follow the base API")

    def rl_step(
        self, action: int, listjobs: Callable[[], list[Job]]
    ) -> list[list[Job]]:
        "Returns a list of jobs for each successful intermediate time step."

        if self.scheduler.step(action):
            return [[]]
        else:
            self.current_time += 1
            try:
                j = self.workload.step()
            except StopIteration as e:
                raise WorkloadExhausted([]) from e
            if j:
                self.scheduler.submit(j)
                self.last_job_time = self.current_time
            self.scheduler.forward_time()
            return [listjobs()]
