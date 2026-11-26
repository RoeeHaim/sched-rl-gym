"""Runtime prediction, the oracle/predicted slot feature and hiding the
actual end of running jobs from the observation."""

import numpy as np
import pytest

from schedgym.envs.compact_env import CompactRmEnv
from schedgym.job import Job
from schedgym.scheduler.null_scheduler import NullScheduler


def _job(jid, user, run, req, procs=1):
    return Job(job_id=jid, submission_time=0, execution_time=run,
               requested_time=req, requested_processors=procs, user_id=user)


def test_prediction_uses_last_two_completed_runs_of_the_user():
    s = NullScheduler(10, 0, ignore_memory=True)
    a = _job(1, 7, run=4, req=20)
    s._submit(a)
    assert a.predicted_time == 20  # no history: requested time
    for jid, run in [(2, 4), (3, 10), (4, 6)]:
        s._user_runtimes.setdefault(7, []).append(run)
        s._user_runtimes[7] = s._user_runtimes[7][-2:]
    b = _job(5, 7, run=1, req=30)
    s._submit(b)
    assert b.predicted_time == 8  # ceil((10 + 6) / 2)
    c = _job(6, 7, run=1, req=5)
    s._submit(c)
    assert c.predicted_time == 5  # never above the request
    d = _job(7, 8, run=1, req=9)
    s._submit(d)
    assert d.predicted_time == 9  # other user: no history


SWF = "\n".join(
    # id submit wait run alloc cpu mem req_procs req_time req_mem status user ...
    f"{i} {i * 3} -1 {run} 2 -1 -1 2 {req} -1 1 {user} -1 -1 -1 -1 -1 -1"
    for i, (run, req, user) in enumerate(
        [(5, 30, 1), (2, 30, 2), (7, 40, 1), (3, 50, 2), (6, 30, 1)] * 8, 1
    )
)


def _env(tmp_path, **kw):
    f = tmp_path / "t.swf"
    f.write_text("; MaxProcs: 4\n" + SWF + "\n")
    return CompactRmEnv(
        processors=4, ignore_memory=True, simulation_type="event_based",
        time_limit=None, job_slots=5,
        workload={"type": "swf", "tracefile": str(f), "restart": True,
                  "cluster_processors": 4},
        **kw,
    )


@pytest.mark.parametrize("feat", [None, "predicted", "oracle"])
def test_observation_size_with_runtime_feature(tmp_path, feat):
    env = _env(tmp_path, runtime_feature=feat, hide_actual_runtime=True)
    obs, _ = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    rng = np.random.default_rng(0)
    for _ in range(60):
        mask = env.action_masks()
        obs, *_ = env.step(int(rng.choice(np.flatnonzero(mask))))
        assert obs.shape == env.observation_space.shape
        assert np.all(np.isfinite(obs))


def test_hidden_runtime_moves_finishes_to_the_requested_end(tmp_path):
    env = _env(tmp_path)
    env.reset(seed=0)
    sched = env.scheduler
    for _ in range(20):
        if sched.queue_running or sched.queue_waiting:
            break
        env.step(0 if env.action_masks()[0] else env.action_space.n - 1)
    job = (sched.queue_running + sched.queue_waiting)[0]

    def finish_offsets(hide):
        state, _, _ = sched.state(env.time_horizon, 5, True, hide)
        times = state[0]
        used = state[1]
        # offset of the first snapshot where the job's processors are free again
        return next(t for t, (free, _u, _d) in zip(times, used) if free == 4 and t > 0)

    real, hidden = finish_offsets(False), finish_offsets(True)
    assert real == job.start_time + job.execution_time - sched.current_time
    assert hidden == job.start_time + job.requested_time - sched.current_time


def test_ewma_ratio_prediction_uses_only_finished_jobs():
    s = NullScheduler(10, 0, ignore_memory=True)
    s.runtime_predictor = "ewma_ratio"
    s.ratio_alpha = 0.3
    first = _job(1, 3, run=5, req=10)
    assert s.predict_runtime(first) == 10  # nothing finished yet
    for jid, (run, req) in enumerate([(5, 10), (2, 10)], 10):
        j = _job(jid, 3, run=run, req=req)
        s.queue_running.append(j)
        j.start_time = 0
        s._complete_job(j)
    # EWMA: 0.5, then 0.3 * 0.2 + 0.7 * 0.5 = 0.41
    assert abs(s._user_ratio[3] - 0.41) < 1e-9
    assert s.predict_runtime(_job(20, 3, run=1, req=100)) == 41
    assert s.predict_runtime(_job(21, 4, run=1, req=100)) == 100  # other user


def test_env_passes_the_predictor_choice(tmp_path):
    env = _env(tmp_path, runtime_feature="predicted", runtime_predictor="ewma_ratio")
    env.reset(seed=0)
    assert env.scheduler.runtime_predictor == "ewma_ratio"
    with pytest.raises(ValueError):
        _env(tmp_path, runtime_predictor="nope")
