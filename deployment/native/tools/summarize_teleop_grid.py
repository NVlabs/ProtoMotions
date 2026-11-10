# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Aggregate a teleop qualification grid (summary.json) into report tables.

    python tools/summarize_teleop_grid.py runs/qualification/summary.json [--html out.html]

Prints one table per factor (arm IK, floor friction, hand mass, push force)
with the upright rate, the median hold error, the median settle time, the
median planted-foot slip and, for pushed cells, the median recovery time.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def median(values):
    values = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return statistics.median(values) if values else None


def fmt(value, digits=1, suffix=""):
    return "n/a" if value is None else f"{value:.{digits}f}{suffix}"


def aggregate(rows, key):
    groups = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)
    table = []
    for value in sorted(groups, key=lambda v: (str(type(v)), v)):
        cells = groups[value]
        upright = sum(1 for c in cells if c.get("upright")) / len(cells)
        recoveries = [r for c in cells for r in (c.get("push_recovery_s") or []) if r is not None]
        timed_out = sum(1 for c in cells for r in (c.get("push_recovery_s") or []) if r is None)
        table.append({
            key: value, "cells": len(cells), "upright_pct": 100 * upright,
            "hold_error_median_cm": median([c.get("hold_error_mean_cm") for c in cells]),
            "hold_error_p95_median_cm": median([c.get("hold_error_p95_cm") for c in cells]),
            "settle_median_s": median([c.get("settle_time_median_s") for c in cells]),
            "slip_median_cm": median([c.get("support_slip_max_cm") for c in cells]),
            "max_tilt_median_deg": median([c.get("max_tilt_deg") for c in cells]),
            "push_recovery_median_s": median(recoveries) if recoveries else None,
            "push_not_recovered": timed_out,
        })
    return table


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary")
    parser.add_argument("--html", default=None)
    args = parser.parse_args()
    rows = json.loads(Path(args.summary).read_text())
    rows = [r for r in rows if r.get("frames")]
    sections = []
    for key, title in (("friction", "Floor friction"), ("hand_mass_kg", "Added hand mass (kg)"), ("push_N", "Torso push (N, 100 ms)")):
        table = aggregate(rows, key)
        sections.append((title, key, table))
        print(f"\n{title}")
        print(f"{'value':>8} {'cells':>5} {'upright':>8} {'hold cm':>8} {'p95 cm':>7} {'settle s':>9} {'slip cm':>8} {'tilt deg':>9} {'recover s':>10} {'no rec':>6}")
        for t in table:
            print(f"{str(t[key]):>8} {t['cells']:>5} {fmt(t['upright_pct'], 0, '%'):>8} {fmt(t['hold_error_median_cm']):>8} {fmt(t['hold_error_p95_median_cm']):>7} "
                  f"{fmt(t['settle_median_s'], 2):>9} {fmt(t['slip_median_cm']):>8} {fmt(t['max_tilt_median_deg']):>9} {fmt(t['push_recovery_median_s'], 2):>10} {t['push_not_recovered']:>6}")
    if args.html:
        out = []
        for title, key, table in sections:
            out.append(f'<h4>{title}</h4><div class="scroll"><table class="data"><thead><tr><th>value</th><th>cells</th><th>upright</th><th>hold error (median of means)</th><th>hold error p95</th><th>settle</th><th>foot slip</th><th>max tilt</th><th>push recovery</th><th>not recovered</th></tr></thead><tbody>')
            for t in table:
                out.append(f"<tr><td>{t[key]}</td><td class=num>{t['cells']}</td><td class=num>{fmt(t['upright_pct'], 0, '%')}</td><td class=num>{fmt(t['hold_error_median_cm'], 1, ' cm')}</td><td class=num>{fmt(t['hold_error_p95_median_cm'], 1, ' cm')}</td>"
                           f"<td class=num>{fmt(t['settle_median_s'], 2, ' s')}</td><td class=num>{fmt(t['slip_median_cm'], 1, ' cm')}</td><td class=num>{fmt(t['max_tilt_median_deg'], 1, '°')}</td><td class=num>{fmt(t['push_recovery_median_s'], 2, ' s')}</td><td class=num>{t['push_not_recovered']}</td></tr>")
            out.append("</tbody></table></div>")
        Path(args.html).write_text("\n".join(out))
        print(f"\nwrote {args.html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
