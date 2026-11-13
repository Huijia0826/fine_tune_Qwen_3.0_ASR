"""Pull audio_ids of onboarding-practice dictations (invocation_context=onboarding_practice) for the probe window.

The cached day scans (online.py COLS) do not carry invocation_context, so it is fetched separately (read-only HogQL).
Usage: python3 scripts/lid_probe/pull_onboarding_ids.py [--since 2026-10-10 --until 2026-11-09]
Writes data/private/lid_probe/onboarding_ids.txt
"""
import argparse
import datetime as dt
import os
import sys
import time

sys.path.insert(0, os.path.expanduser("~/.claude/skills/trace-lookup/scripts"))
from trace_lookup import POSTHOG, load_env, q  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--since", default="2026-10-10")
ap.add_argument("--until", default="2026-11-09")
a = ap.parse_args()
load_env(POSTHOG)
ids, d = set(), dt.date.fromisoformat(a.since)
while d < dt.date.fromisoformat(a.until):
    e = d + dt.timedelta(days=1)
    for attempt in range(5):  # the local proxy drops connections now and then
        try:
            res = q("SELECT DISTINCT properties.audio_id FROM events WHERE event='asr_metrics' "
                    f"AND timestamp >= '{d}' AND timestamp < '{e}' AND properties.invocation_context = 'onboarding_practice' LIMIT 50000")
            break
        except Exception as exc:  # noqa: BLE001
            print(f"{d}: retry {attempt + 1} ({type(exc).__name__})", file=sys.stderr)
            time.sleep(5)
    else:
        raise SystemExit(f"PostHog failed for {d}")
    ids |= {r[0] for r in res if r[0]}
    d = e
out = "data/private/lid_probe/onboarding_ids.txt"
open(out, "w").write("\n".join(sorted(ids)) + "\n")
print(f"{len(ids)} onboarding audio_ids -> {out}")
