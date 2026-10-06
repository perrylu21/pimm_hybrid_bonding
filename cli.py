#!/usr/bin/env python3
"""Command-line entry point for the hybrid-bonding die-shift study.

  python cli.py run studies/pilot.yaml [--retry-failed]
  python cli.py collect results/pilot
"""
import argparse


def cmd_run(a):
    from hbsim import params, runner
    runner.run_study(params.load_study(a.study), retry_failed=a.retry_failed)


def cmd_collect(a):
    from hbsim import runner
    df = runner.collect(a.study_dir)
    print(df["status"].value_counts().to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(required=True)
    p = sub.add_parser("run", help="execute a study YAML")
    p.add_argument("study")
    p.add_argument("--retry-failed", action="store_true")
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("collect", help="rebuild dataset.parquet for a study directory")
    p.add_argument("study_dir")
    p.set_defaults(fn=cmd_collect)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
