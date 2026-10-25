"""trace_demo.py - see what happens inside the simulator, step by step.

Run:  python trace_demo.py
Tiny cluster (10 processors), three hand-made jobs, and we play the "agent"
by hand, printing the full internal state after every call.
"""
import warnings
warnings.filterwarnings("ignore")

from schedgym.job import Job
from schedgym.scheduler import NullScheduler

NAMES = {1: "A", 2: "B", 3: "C"}


def make_job(jid, procs, length):
    j = Job(jid, 0, execution_time=length, processors_allocated=procs,
            requested_processors=procs, requested_time=length)
    j.requested_memory = 0
    return j


def show(s, title):
    print(f"\n=== {title} ===")
    print(f"clock: current_time = {s.current_time}")
    cpus = ["."] * s.number_of_processors
    for j in s.queue_running:
        for iv in j.resources.processors:
            for p in range(iv.begin, iv.end):
                cpus[p] = NAMES[j.id]
    print(f"processors 0..9 : [{''.join(cpus)}]   (. = free)")
    q = lambda lst: [NAMES[j.id] for j in lst]
    print(f"queue_admission (waiting for a decision): {q(s.queue_admission)}")
    print(f"queue_waiting   (got a start time)      : {q(s.queue_waiting)}")
    print(f"queue_running   (running)               : {q(s.queue_running)}")
    print(f"queue_completed (finished)              : {q(s.queue_completed)}")
    evs = [f"t={e.time}:{e.type.name}({NAMES[e.job.id]})" for e in s.job_events]
    print(f"future events: {evs if evs else '(none)'}")


s = NullScheduler(number_of_processors=10, total_memory=10, ignore_memory=True)
A, B, C = make_job(1, 6, 3), make_job(2, 6, 2), make_job(3, 2, 1)
show(s, "empty cluster")

s.submit([A, B, C])
show(s, "3 jobs arrived: A=6 procs x 3 time, B=6x2, C=2x1")

ok = s.step(0)
show(s, f"agent: 'start slot 0 (A)' -> {ok}")

ok = s.step(0)
show(s, f"agent: 'start slot 0 (now B)' -> {ok}  (only 4 procs left)")

ok = s.step(1)
show(s, f"agent: 'start slot 1 (C)' -> {ok}")

for _ in range(6):
    s.forward_time()
    show(s, "agent: 'do nothing' -> forward_time() (clock +1)")
    if s.queue_admission:
        ok = s.step(0)
        show(s, f"agent: 'start slot 0 ({NAMES[s.queue_admission[0].id] if not ok else 'B'})' -> {ok}")

print("\nresults:")
for j in (A, B, C):
    print(f"  {NAMES[j.id]}: submitted={j.submission_time} start={j.start_time} "
          f"finish={j.finish_time} wait={j.wait_time} slowdown={j.slowdown:.2f}")
