#
import argparse
from typing import Dict
from apps.miod.miod_app import MiodApp

def main(params:Dict = {}) -> None:
    MiodApp.startup(params=params)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--run_mode', action='store',
        type=int, default=1, dest='run_mode',
        help='run mode'
    )
    return parser.parse_args()

if '__main__' == __name__:
    args = parse_args()
    params = vars(args)
    main(params=params)