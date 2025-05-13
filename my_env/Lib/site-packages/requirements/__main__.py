#! /usr/bin/env python3

"""Find updates for packages in requirements.txt on pypi"""

__all__ = []

import os
import sys
from .requirements import verbose


def main():
    verbose(sys.argv[1:])


# If we are running from a wheel, add the wheel to sys.path
if __package__ == '':
    path = os.path.dirname(os.path.dirname(__file__))
    sys.path.insert(0, path)

if __name__ == "__main__":
    main()
