#!/usr/bin/env python3
"""Quick progress check for the baseline pipeline.

Reads baseline_citations.json and baseline.log to show real-time
progress, speed, ETA, and error count.

Usage:
    python3.11 baseline/check_progress.py
    watch -n 5 "python3.11 baseline/check_progress.py"
"""
import json
import os
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

BASELINE_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = BASELINE_DIR / "baseline_citations.json"
LOG_FILE = BASELINE_DIR / "baseline.log"
FAILED_FILE = BASELINE_DIR / "baseline_failed.json"
CACHE_DIR = BASELINE_DIR / "parser_cache"

# Timestamp pattern: "2026-04-22 17:11:07,505 [INFO] ..."
_TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})\s")


def _parse_ts(s: str) -> datetime | None:
    try:
        return datetime.strptime(s, "%Y-%m-%d %H:%M:%S,%f")
    except ValueError:
        return None


def _fmt_td(td: timedelta) -> str:
    total = int(td.total_seconds())
    if total < 0:
        return "—"
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    elif m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def main():
    # ── 1. Count total papers in cache ──
    total_papers = len(list(CACHE_DIR.glob("PMC*.json")))

    # ── 2. Read output file ──
    n_done = 0
    total_sw = 0
    if OUTPUT_FILE.exists():
        try:
            with open(OUTPUT_FILE) as f:
                results = json.load(f)
            n_done = len(results)
            total_sw = sum(len(r.get("software_citations", [])) for r in results)
        except (json.JSONDecodeError, KeyError):
            pass

    # ── 3. Read log for timestamps and stats ──
    start_time = None
    last_time = None
    error_count = 0
    recent_pubs = []

    if LOG_FILE.exists():
        first_ts = None
        last_ts_str = None

        with open(LOG_FILE) as f:
            for line in f:
                m = _TS_RE.match(line)
                if m:
                    ts_str = m.group(1)
                    if first_ts is None:
                        first_ts = ts_str
                    last_ts_str = ts_str

                if "[ERROR]" in line:
                    error_count += 1

                # Track recent papers
                if "citations built" in line:
                    pub_match = re.search(r"(PMC\d+):\s*(\d+)\s*citations built in\s*([\d.]+)s", line)
                    if pub_match:
                        recent_pubs.append((
                            pub_match.group(1),
                            int(pub_match.group(2)),
                            float(pub_match.group(3)),
                        ))

        if first_ts:
            start_time = _parse_ts(first_ts)
        if last_ts_str:
            last_time = _parse_ts(last_ts_str)

    # Fallback timestamps
    if start_time is None and LOG_FILE.exists():
        start_time = datetime.fromtimestamp(os.path.getmtime(LOG_FILE))

    now = datetime.now()
    elapsed = now - start_time if start_time else timedelta(0)

    # ── 4. Failed papers ──
    n_failed = 0
    if FAILED_FILE.exists():
        try:
            with open(FAILED_FILE) as f:
                n_failed = len(json.load(f))
        except (json.JSONDecodeError, KeyError):
            pass

    # ── 5. Calculate speed & ETA ──
    remaining = max(total_papers - n_done, 0)
    if n_done > 0 and elapsed.total_seconds() > 0:
        secs_per_paper = elapsed.total_seconds() / n_done
        eta_secs = secs_per_paper * remaining
        eta = timedelta(seconds=int(eta_secs))
        speed_str = f"{secs_per_paper:.1f}s/paper ({60/secs_per_paper:.1f} papers/min)"
    else:
        eta = timedelta(0)
        speed_str = "calculating..."

    # ── 6. Print report ──
    last_activity = last_time.strftime('%H:%M:%S') if last_time else "—"
    pct = (n_done / total_papers * 100) if total_papers else 0
    bar_width = 30
    filled = min(int(bar_width * n_done / total_papers) if total_papers else 0, bar_width)
    bar = "█" * filled + "░" * (bar_width - filled)

    print(f"📊 Baseline Pipeline Progress")
    print(f"{'─' * 50}")
    print(f"  Progress:     [{bar}] {pct:.1f}%")
    print(f"  Papers done:  {n_done}/{total_papers}")
    print(f"  Software:     {total_sw} mentions extracted")
    print(f"  Avg SW/paper: {total_sw/n_done:.1f}" if n_done > 0 else "  Avg SW/paper: —")
    print(f"  Elapsed:      {_fmt_td(elapsed)}")
    print(f"  Speed:        {speed_str}")

    if remaining > 0:
        print(f"  Remaining:    {remaining} papers")
        print(f"  ETA:          {_fmt_td(eta)}")
    else:
        print(f"  ✅ All papers processed!")

    print(f"{'─' * 50}")
    print(f"  Errors:       {error_count} (log) / {n_failed} (failed file)")
    print(f"  Last activity: {last_activity}")

    if OUTPUT_FILE.exists():
        cp_mtime = datetime.fromtimestamp(os.path.getmtime(OUTPUT_FILE))
        print(f"  Output saved: {cp_mtime.strftime('%H:%M:%S')}")

    # Show recent papers
    if recent_pubs:
        last_few = recent_pubs[-5:]
        print(f"{'─' * 50}")
        print(f"  Recent papers:")
        for pid, n_sw, secs in last_few:
            print(f"    {pid}: {n_sw} software in {secs:.1f}s")


if __name__ == "__main__":
    main()
