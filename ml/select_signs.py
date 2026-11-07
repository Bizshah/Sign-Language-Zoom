"""Pick ~50 target signs from WLASL and check each has enough usable videos.

Usage:
  python ml/select_signs.py --meta WLASL_v0.3.json --videos data/videos \
      --min-samples 20 --out ml/signs.json
"""
import argparse
import json
from pathlib import Path

# Meeting-oriented seed list. Extend / edit freely; missing glosses are reported.
SEED = [
    "hello", "thank you", "yes", "no", "please", "help", "question",
    "understand", "repeat", "wait", "meeting", "name", "good", "bad",
    "sorry", "again", "slow", "fast", "stop", "start", "finish", "like",
    "want", "need", "know", "think", "work", "time", "today", "tomorrow",
    "yesterday", "now", "later", "who", "what", "where", "when", "why",
    "how", "read", "write", "show", "learn", "teach", "ready", "agree",
    "different", "same", "more", "all",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True)
    ap.add_argument("--videos", required=True)
    ap.add_argument("--min-samples", type=int, default=20)
    ap.add_argument("--n-signs", type=int, default=50)
    ap.add_argument("--out", default="ml/signs.json")
    args = ap.parse_args()

    video_dir = Path(args.videos)
    meta = json.load(open(args.meta))

    # gloss -> list of {video_id, split}; keep only videos that exist on disk
    counts = {}
    for entry in meta:
        g = entry["gloss"].lower()
        ok = [
            {"video_id": i["video_id"], "split": i.get("split", "train")}
            for i in entry["instances"]
            if (video_dir / f"{i['video_id']}.mp4").exists()
        ]
        counts[g] = ok

    print(f"{'gloss':<14}{'usable':>7}")
    chosen, dropped, missing = [], [], []
    for g in SEED:
        n = len(counts.get(g, []))
        if g not in counts:
            missing.append(g)
        elif n >= args.min_samples:
            chosen.append(g)
            print(f"{g:<14}{n:>7}  OK")
        else:
            dropped.append((g, n))
            print(f"{g:<14}{n:>7}  too few")

    # Top up with the best-populated remaining glosses if we're short of 50
    if len(chosen) < args.n_signs:
        rest = sorted(
            (g for g in counts if g not in chosen and g not in SEED),
            key=lambda g: -len(counts[g]),
        )
        for g in rest:
            if len(chosen) >= args.n_signs:
                break
            if len(counts[g]) >= args.min_samples:
                chosen.append(g)

    chosen = chosen[: args.n_signs]
    print(f"\nSelected {len(chosen)} signs.")
    if missing:
        print("Not in dataset:", ", ".join(missing))
    if dropped:
        print("Dropped (too few samples):", ", ".join(f"{g}({n})" for g, n in dropped))
    if len(chosen) < args.n_signs:
        print(f"Only {len(chosen)} signs reach {args.min_samples}+ samples. "
              f"Re-run with a lower --min-samples (e.g. 10-12) if you need 50.")

    out = {g: counts[g] for g in chosen}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()