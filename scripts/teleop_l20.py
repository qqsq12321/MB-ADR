"""Source-checkout wrapper for the installed mbadr-l20 command."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mbadr.teleop_l20 import build_parser, main, validate_checkpoint

if __name__ == "__main__":
    main()
