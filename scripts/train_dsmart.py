"""Train / evaluate a DSMART agent on sched-rl-gym (continuing regime).

Examples::

    python scripts/train_dsmart.py --set algo=DS time_budget=50000 log_path=runs/ds.jsonl
    python scripts/train_dsmart.py --config cfg.json --set seed=3
    python scripts/train_dsmart.py --set algo=SJF          # heuristic reference only

``--set`` values are parsed as JSON when possible (numbers, true/false, lists),
otherwise taken as strings; dotted keys reach into dicts, e.g.
``env_kwargs.workload.new_job_rate=0.4``. See ``schedgym/agents/dsmart/train.py``
for every key and its default.
"""

import argparse
import json

from schedgym.agents.dsmart.train import run


def _parse(v):
    try:
        return json.loads(v)
    except json.JSONDecodeError:
        return v


def _set(cfg, dotted, value):
    keys = dotted.split(".")
    d = cfg
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="JSON file with config overrides")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = p.parse_args()
    cfg = {}
    if args.config:
        with open(args.config) as f:
            cfg = json.load(f)
    for kv in args.set:
        k, v = kv.split("=", 1)
        _set(cfg, k, _parse(v))
    final = run(cfg)
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
