from schedgym.job import Job
from schedgym.workload.trace import TraceGenerator


def _jobs():
    return [
        Job(job_id=i, submission_time=t, execution_time=2, requested_time=2,
            requested_processors=1)
        for i, t in enumerate([5, 7, 10])
    ]


def test_restart_laps_have_constant_period():
    g = TraceGenerator(restart=True, trace=_jobs())
    g.current_time = 4
    seen = []
    for _ in range(40):
        seen += g.step()
    times = [j.submission_time for j in seen]
    assert times == sorted(times)
    # every lap starts right after the previous one: no growing idle gap
    assert max(b - a for a, b in zip(times, times[1:])) <= 3
    assert len(seen) >= 18


def test_restart_uses_fresh_job_objects():
    g = TraceGenerator(restart=True, trace=_jobs())
    g.current_time = 4
    first = []
    while len(first) < 3:
        first += g.step()
    first[0].start_time = 123  # simulate run state left on the job
    second = []
    while len(second) < 1:
        second += g.step()
    assert second[0] is not first[0]
    assert second[0].id == first[0].id
