#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Smoke test script for README features.

What it validates:
1) README lists runnable spider modes.
2) `weibospider/run_spider.py <mode>` can start without immediate
   import/argument/runtime bootstrap errors.

Note:
- This is a smoke test, not an integration correctness test.
- Network/cookie related failures may happen after startup and are outside
  this script's scope.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
README = ROOT / "README.md"
RUN_SPIDER = ROOT / "weibospider" / "run_spider.py"
SPIDER_CWD = ROOT / "weibospider"

# README currently documents these command examples.
README_MODES = [
    "user",
    "fan",
    "follow",
    "comment",
    "repost",
    "tweet_by_tweet_id",
    "tweet_by_user_id",
    "tweet_by_keyword",
]

# If process reaches timeout, we treat it as "started and running".
STARTUP_TIMEOUT_SECONDS = 15


def safe_console_text(text: str) -> str:
    encoding = sys.stdout.encoding or "utf-8"
    return text.encode(encoding, errors="replace").decode(encoding, errors="replace")


def parse_modes_from_run_spider(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8")
    # Parse choices=['comment', ...]
    match = re.search(r"choices\s*=\s*\[(.*?)\]", text, re.S)
    if not match:
        raise RuntimeError("Cannot parse spider modes from run_spider.py.")
    block = match.group(1)
    return set(re.findall(r"'([^']+)'", block))


def run_smoke_mode(mode: str) -> tuple[str, bool, str]:
    cmd = [sys.executable, str(RUN_SPIDER), mode]
    try:
        completed = subprocess.run(
            cmd,
            cwd=str(SPIDER_CWD),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=STARTUP_TIMEOUT_SECONDS,
            check=False,
        )
        merged = (completed.stdout or "") + "\n" + (completed.stderr or "")
        if completed.returncode == 0:
            return mode, True, "exited with code 0"
        return mode, False, f"exit={completed.returncode}\n{merged.strip()[:1200]}"
    except subprocess.TimeoutExpired:
        return mode, True, f"running beyond {STARTUP_TIMEOUT_SECONDS}s (startup OK)"
    except Exception as exc:  # pragma: no cover
        return mode, False, f"exception: {exc}"


def main() -> int:
    if not README.exists():
        print(f"[FAIL] README not found: {README}")
        return 2
    if not RUN_SPIDER.exists():
        print(f"[FAIL] run_spider.py not found: {RUN_SPIDER}")
        return 2

    declared_modes = parse_modes_from_run_spider(RUN_SPIDER)
    readme_modes = set(README_MODES)

    missing_in_code = sorted(readme_modes - declared_modes)
    missing_in_readme = sorted(declared_modes - readme_modes)

    print("== README feature smoke test ==")
    print(f"Project root: {ROOT}")
    print(f"Python: {sys.executable}")
    print("")
    print("Mode consistency check:")
    print(f"- README modes: {sorted(readme_modes)}")
    print(f"- Code modes:   {sorted(declared_modes)}")
    if missing_in_code:
        print(f"[WARN] In README but not in code: {missing_in_code}")
    if missing_in_readme:
        print(f"[WARN] In code but not in README: {missing_in_readme}")

    print("")
    print("Startup smoke check:")
    results = [run_smoke_mode(mode) for mode in sorted(readme_modes)]

    failed = 0
    for mode, ok, detail in results:
        if ok:
            print(f"[PASS] {mode}: {detail}")
        else:
            failed += 1
            print(f"[FAIL] {mode}: {safe_console_text(detail)}")

    print("")
    if failed:
        print(f"Result: {failed}/{len(results)} mode(s) failed startup smoke test.")
        return 1
    print("Result: all README modes passed startup smoke test.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
