"""Source-checkout wrapper for mbadr-pico-relay."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mbadr.pico_relay import main

if __name__ == "__main__":
    main()
