import sys

from config import SITE_PATH, CONC_PATH, WIND_PATH
from pipeline import run


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except AttributeError:
        pass
    run(SITE_PATH, CONC_PATH, WIND_PATH)
