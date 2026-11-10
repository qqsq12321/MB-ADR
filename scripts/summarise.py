"""Aggregate per-hand evaluation JSON into one comparison table.

Reads ``out/eval_<robot>.json`` written by ``mbadr.evaluate`` and prints a
single table, plus a markdown block that can be pasted into the README.

    python scripts/summarise.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mbadr.config import OUT_DIR, SUPPORTED_HANDS  # noqa: E402

HANDS = ["l20", "shadow", "wuji", "sharpa", "gaia", "inspire"]


def load(robot: str) -> dict | None:
    path = OUT_DIR / f"eval_{robot}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    # evaluate.py keys results by recording name; take the only entry.
    return next(iter(data.values()))


def main() -> None:
    rows = []
    for robot in HANDS:
        r = load(robot)
        if r is None:
            rows.append((robot, None))
            continue
        rows.append((robot, r))

    header = (
        f"{'hand':9s} {'frames':>7s} {'pinch':>6s} "
        f"{'ratio':>7s} {'tip net':>8s} {'tip nlopt':>10s} "
        f"{'gap net':>8s} {'gap nlopt':>10s} {'latency':>9s}"
    )
    print(header)
    print("-" * len(header))
    for robot, r in rows:
        if r is None:
            print(f"{robot:9s} {'(not evaluated)':>45s}")
            continue
        pinch = r["pinch_gap_cm"]
        net_gap = pinch.get("net_pinch") or {}
        ref_gap = pinch.get("nlopt_pinch") or {}
        print(
            f"{robot:9s} {r['frames']:7d} {r['pinch_active_frames']:6d} "
            f"{r['loss_ratio']:7.3f} "
            f"{r['fingertip_err_cm']['net']['mean']:8.4f} "
            f"{r['fingertip_err_cm']['nlopt']['mean']:10.4f} "
            f"{net_gap.get('mean', float('nan')):8.4f} "
            f"{ref_gap.get('mean', float('nan')):10.4f} "
            f"{r['latency_ms']['net'] * 1000:8.0f}us"
        )

    print("\nmarkdown:")
    print(
        "| hand | frames | pinch frames | loss ratio | tip err net | tip err NLopt "
        "| gap net | gap NLopt | latency |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    for robot, r in rows:
        if r is None:
            continue
        pinch = r["pinch_gap_cm"]
        net_gap = pinch.get("net_pinch") or {}
        ref_gap = pinch.get("nlopt_pinch") or {}
        print(
            f"| `{robot}` | {r['frames']} | {r['pinch_active_frames']} "
            f"| {r['loss_ratio']:.3f} "
            f"| {r['fingertip_err_cm']['net']['mean']:.2f} cm "
            f"| {r['fingertip_err_cm']['nlopt']['mean']:.2f} cm "
            f"| {net_gap.get('mean', float('nan')):.2f} cm "
            f"| {ref_gap.get('mean', float('nan')):.2f} cm "
            f"| {r['latency_ms']['net'] * 1000:.0f} us |"
        )


if __name__ == "__main__":
    main()
