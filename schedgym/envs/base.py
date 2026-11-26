from abc import ABC, abstractmethod
from enum import IntEnum

import gymnasium
import numpy as np

from ..scheduler.null_scheduler import NullScheduler
from .simulator import DeepRmSimulator, SimulationType
from .workload import build as build_workload

BACKLOG_SIZE = 60
MAXIMUM_NUMBER_OF_ACTIVE_JOBS = 40  # Number of colors in image
MAX_TIME_TRACKING_SINCE_LAST_JOB = 10

TIME_HORIZON = 20
TIME_LIMIT = 200
JOB_SLOTS = 5
AMOUNT_OF_MEMORY = 10
NUMBER_OF_PROCESSORS = 10
MAXIMUM_JOB_LENGTH = 15
MAXIMUM_JOB_SIZE = 10
NEW_JOB_RATE = 0.7
SMALL_JOB_CHANCE = 0.8
DEFAULT_WORKLOAD = {
    "type": "deeprm",
    "new_job_rate": NEW_JOB_RATE,
    "max_job_size": MAXIMUM_JOB_SIZE,
    "max_job_len": MAXIMUM_JOB_LENGTH,
    "small_job_chance": SMALL_JOB_CHANCE,
}


class RewardJobs(IntEnum):
    ALL = 0
    JOB_SLOTS = 1
    WAITING = 2
    RUNNING_JOB_SLOTS = 3

    @staticmethod
    def from_str(reward_range: str):
        reward_range = reward_range.upper().replace("-", "_")
        if reward_range in RewardJobs.__members__:
            return RewardJobs[reward_range]
        else:
            raise ValueError(
                f"{reward_range} is not a valid RewardJobs range. "
                f"Valid options are: {list(RewardJobs.__members__.keys())}."
            )


class BaseRmEnv(ABC, gymnasium.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    job_slots: int
    time_limit: int
    job_num_cap: int
    time_horizon: int
    ignore_memory: bool
    color_index: list[int]
    color_cache: dict[int, int]
    simulator: DeepRmSimulator

    @abstractmethod
    def __init__(self, **kwargs):
        self.color_cache = {}
        self.renderer = kwargs.get("renderer", None)
        self.shuffle_colors = kwargs.get("shuffle_colors", False)
        self.job_num_cap = kwargs.get("job_num_cap", MAXIMUM_NUMBER_OF_ACTIVE_JOBS)
        self.simulation_type = SimulationType.from_str(
            kwargs.get("simulation_type", "time_based")
        )

        self.reward_jobs = RewardJobs.from_str(kwargs.get("reward_jobs", "all"))
        # How long "wait" lasts in the event-based simulator when a job already
        # fits: "unit" (one time unit) or "next_event" (until a job arrives or
        # finishes).
        self.wait_mode = kwargs.get("wait_mode", "unit")

        self.smdp = self.simulation_type == SimulationType.EVENT_BASED
        # No discount factor here: discounting (or averaging) is the learner's
        # job. Each step reports raw per-time-unit rewards in
        # info["step_rewards"] and the elapsed time in info["duration"].

        self.time_horizon = kwargs.get(
            "time_horizon", TIME_HORIZON
        )  # number of time steps in the graph

        self.tolerance_factor = kwargs.get("tolerance_factor", 100)

        # time_limit=None means "no time limit": a trace workload derives one
        # from the trace at reset (tolerance_factor * end of trace); any other
        # workload never truncates (continuing task). The observation still
        # needs a fixed time scale to normalise by, which is ``obs_time_scale``.
        time_limit = kwargs.get("time_limit", TIME_LIMIT)
        if time_limit is None:
            self.time_limit = kwargs.get("obs_time_scale", TIME_LIMIT)
            self.update_time_limit = True
            self.truncate = False
        else:
            self.time_limit = time_limit
            self.update_time_limit = False
            self.truncate = bool(time_limit)

        step = 1.0 / self.job_num_cap
        # zero is already present and set to "no job there"
        self.colormap = np.arange(start=step, stop=1, step=step)
        if self.shuffle_colors:
            np.random.shuffle(self.colormap)
        self.color_index = list(range(len(self.colormap)))

        # Number of jobs to show
        self.job_slots = kwargs.get("job_slots", JOB_SLOTS)

        self.reward_mapper = {
            RewardJobs.ALL: lambda: self.scheduler.jobs_in_system,
            RewardJobs.WAITING: lambda: self.scheduler.queue_admission,
            RewardJobs.JOB_SLOTS: lambda: self.scheduler.queue_admission[
                : self.job_slots
            ],
            RewardJobs.RUNNING_JOB_SLOTS: lambda: (
                self.scheduler.queue_running
                + self.scheduler.queue_admission[: self.job_slots]
            ),
        }

        self.backlog_size = kwargs.get("backlog_size", BACKLOG_SIZE)
        self.memory = kwargs.get("memory", AMOUNT_OF_MEMORY)
        self.processors = kwargs.get("processors", NUMBER_OF_PROCESSORS)
        self.ignore_memory = kwargs.get("ignore_memory", False)
        # Runtime predictor behind predicted runtimes ("tsafrir" or
        # "ewma_ratio") and the EWMA rate of the latter.
        self.runtime_predictor = kwargs.get("runtime_predictor", "tsafrir")
        if self.runtime_predictor not in ("tsafrir", "ewma_ratio"):
            raise ValueError(
                "runtime_predictor must be 'tsafrir' or 'ewma_ratio', "
                f"not {self.runtime_predictor!r}"
            )
        self.ratio_alpha = kwargs.get("ratio_alpha", 0.3)

        self.workload_config = kwargs.get("workload", DEFAULT_WORKLOAD)
        wl = build_workload(self.workload_config)

        scheduler = NullScheduler(
            self.processors, self.memory, ignore_memory=self.ignore_memory
        )
        self._configure_predictor(scheduler)
        self.simulator = DeepRmSimulator(
            wl,
            scheduler,
            simulation_type=self.simulation_type,
            job_slots=self.job_slots,
            wait_mode=self.wait_mode,
        )

    def _configure_predictor(self, scheduler) -> None:
        scheduler.runtime_predictor = self.runtime_predictor
        scheduler.ratio_alpha = self.ratio_alpha

    def reset(self, *, seed=None, options=None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed, options=options)
        scheduler = NullScheduler(
            self.processors, self.memory, ignore_memory=self.ignore_memory
        )
        self._configure_predictor(scheduler)
        wl = build_workload(self.workload_config)
        if self.update_time_limit and hasattr(wl, "trace") and wl.trace:
            self.time_limit = self.tolerance_factor * (
                wl.trace[-1].submission_time  # type: ignore
                + wl.trace[-1].execution_time  # type: ignore
            )
            self.truncate = True
        self.simulator.reset(wl, scheduler)
        return self.state, {}

    def _render_state(self):
        state, jobs, backlog = self.scheduler.state(self.time_horizon, self.job_slots)
        s = self._convert_state(
            state,
            jobs,
            backlog,
            (
                (self.simulator.current_time - self.simulator.last_job_time)
                / MAX_TIME_TRACKING_SINCE_LAST_JOB
            ),
        )
        return s

    def build_current_state(self, current):
        ret = [np.zeros((self.time_horizon, sum(e[0][:-1]))) for e in current]
        for i, _ in enumerate(current):
            for t, c in enumerate(current[i]):
                for k, v in c[-1].items():
                    ret[i][t][slice(*k)] = v
        return ret

    def build_job_slots(self, wait):
        memory = np.zeros(
            (self.job_slots, self.time_horizon, self.scheduler.total_memory)
        )
        processors = np.zeros(
            (
                self.job_slots,
                self.time_horizon,
                self.scheduler.number_of_processors,
            )
        )
        for i, j in enumerate(wait):
            if j.requested_processors == -1:
                break
            time_slice = slice(
                0,
                self.time_horizon
                if j.requested_time > self.time_horizon
                else j.requested_time,
            )
            processors[i, time_slice, : j.requested_processors] = 1.0
            if j.requested_memory != -1:
                memory[i, time_slice, : j.requested_memory] = 1.0
        return (processors,) if self.ignore_memory else (processors, memory)

    def _convert_state(self, current, wait, backlog, time):
        current = self.build_current_state(current[1:] if self.smdp else current)
        wait = self.build_job_slots(wait)
        backlog_width = self.backlog_size // self.time_horizon
        backlog = np.ones(self.time_horizon * backlog_width) * backlog
        unique = set(np.unique(current[0])) - {0.0}
        if len(unique) > self.job_num_cap:
            raise AssertionError("Number of jobs > number of colors")
        available_colors = list(
            set(self.color_index)
            - set([self.color_cache[j] for j in unique if j in self.color_cache])
        )
        need_color = unique - set(self.color_cache.keys())
        for i, j in enumerate(need_color):
            self.color_cache[j] = available_colors[i]
        max_job_id = int(max(unique)) if unique else 0
        lut = np.zeros(max_job_id + 1, dtype=np.float64)
        for j in unique:
            lut[int(j)] = self.colormap[self.color_cache[j]]
        for resource in current:
            resource[:] = lut[resource.astype(np.intp)]

        return (
            np.array(current),
            np.array(wait),
            backlog.reshape((self.time_horizon, -1)),
            np.ones((self.time_horizon, 1)) * min(1.0, time),
        )

    def render(self):
        if self.renderer is None:
            from .render import DeepRmRenderer

            self.renderer = DeepRmRenderer(self.render_mode or "human")
        rgb = self.renderer.render(self._render_state())
        return rgb

    def compute_reward(self, joblist):
        return -np.sum([1 / j.execution_time for j in joblist if j.execution_time > 0])

    def action_masks(self) -> np.ndarray:
        """Valid actions right now, for MaskablePPO.

        Action i (i < job_slots) is valid only if the i-th job in the queue
        exists and can start now; picking a job that does not fit would just
        behave like waiting. The last action ("wait") is always valid.
        """
        mask = np.zeros(self.action_space.n, dtype=bool)
        for i, job in enumerate(self.scheduler.queue_admission[: self.job_slots]):
            mask[i] = bool(self.scheduler.can_schedule_now(job))
        mask[-1] = True
        return mask

    def _transition_rewards(self, intermediate: list, duration: int) -> tuple:
        """Raw (undiscounted) rewards of one decision-to-decision transition.

        Returns ``(reward, step_rewards)``: ``step_rewards[k]`` is the reward of
        the k-th elapsed time unit and ``reward == sum(step_rewards)``. A
        transition that does not advance time (a job started at once) has no
        time units, so it returns ``(0.0, [])``.
        """
        if duration <= 0:
            return 0.0, []
        step_rewards = [float(self.compute_reward(jobs)) for jobs in intermediate]
        return float(np.sum(step_rewards)), step_rewards

    @property
    def reward(self):
        return self.compute_reward(self.reward_mapper[self.reward_jobs]())

    @property
    def stats(self):
        return self.scheduler.stats

    @property
    @abstractmethod
    def state(self):
        raise NotImplementedError

    @property
    def scheduler(self) -> NullScheduler:
        return self.simulator.scheduler
