"""Run an external policy through the same API used by the installed CLI."""
import argparse
import json

from custom_policy import make_policy
from vision_arm_lab import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    report = evaluate(args.config, policy_factory=make_policy, seeds=[0], steps=20)
    print(json.dumps(report.summary, indent=2))


if __name__ == '__main__':
    main()
