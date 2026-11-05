"""
ml/inspect_dataset.py

Quick inspection of a downloaded sign-language dataset, so you can fill in
docs/data_notes.md with real numbers.

Usage:
    # Google ISLR (Kaggle asl-signs), folder that contains train.csv
    python ml/inspect_dataset.py islr data/asl-signs

    # INCLUDE (Zenodo), folder that contains the category folders of videos
    python ml/inspect_dataset.py include data/include

Prints: number of signs, samples per sign (min / median / max), number of
signers (ISLR), file format details, and any missing files.
"""

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd


def describe_counts(counts: pd.Series, label: str):
    print(f"\n--- Samples per {label} ---")
    print(f"number of {label}s : {counts.shape[0]}")
    print(f"total samples     : {int(counts.sum())}")
    print(f"min / median / max: {int(counts.min())} / {int(counts.median())} / {int(counts.max())}")
    print(f"mean              : {counts.mean():.1f}")
    print("\nFewest samples:")
    print(counts.nsmallest(5).to_string())
    print("\nMost samples:")
    print(counts.nlargest(5).to_string())


def inspect_islr(root: Path):
    csv_path = root / "train.csv"
    if not csv_path.exists():
        raise SystemExit(f"train.csv not found in {root}")

    df = pd.read_csv(csv_path)
    print("train.csv columns:", list(df.columns))
    print("rows (sequences) :", len(df))

    describe_counts(df["sign"].value_counts(), "sign")

    if "participant_id" in df.columns:
        print(f"\nnumber of signers (participant_id): {df['participant_id'].nunique()}")
        print("sequences per signer (min / median / max):",
              df["participant_id"].value_counts().agg(["min", "median", "max"]).astype(int).tolist())

    # How many of the files listed in train.csv are actually on disk?
    exists = df["path"].apply(lambda p: (root / p).exists())
    print(f"\nlandmark files on disk: {int(exists.sum())} / {len(df)}")

    # Look at one parquet file in detail
    on_disk = df[exists]
    if on_disk.empty:
        print("No parquet files downloaded yet, skipping file-format check.")
        return
    sample_path = root / on_disk.iloc[0]["path"]
    pq = pd.read_parquet(sample_path)
    print(f"\n--- One sample file: {sample_path.name} ---")
    print("columns :", list(pq.columns))
    print("rows    :", len(pq))
    if "frame" in pq.columns:
        print("frames  :", pq["frame"].nunique())
    if "type" in pq.columns:
        print("landmarks per type (unique landmark_index):")
        print(pq.groupby("type")["landmark_index"].nunique().to_string())
        print("fraction of NaN x per type (missing detections):")
        print(pq.groupby("type")["x"].apply(lambda s: s.isna().mean()).round(3).to_string())


def inspect_include(root: Path):
    exts = {".mov", ".mp4", ".avi", ".mkv"}
    videos = [p for p in root.rglob("*") if p.suffix.lower() in exts]
    if not videos:
        raise SystemExit(f"No video files found under {root}")

    # Expected layout: <root>/<Category>/<N. Sign>/<video file>
    sign_counter = Counter()
    category_counter = Counter()
    for v in videos:
        rel = v.relative_to(root).parts
        if len(rel) >= 3:
            category_counter[rel[-3]] += 1
            sign_counter[f"{rel[-3]}/{rel[-2]}"] += 1
        else:
            sign_counter[v.parent.name] += 1

    counts = pd.Series(sign_counter)
    describe_counts(counts, "sign")
    print(f"\nnumber of categories: {len(category_counter)}")
    print(pd.Series(category_counter).sort_values(ascending=False).to_string())
    print("\nvideo file extensions:", Counter(v.suffix.lower() for v in videos))

    # Peek at one video's properties
    try:
        import cv2
        cap = cv2.VideoCapture(str(videos[0]))
        print(f"\n--- One sample video: {videos[0].name} ---")
        print("resolution:", int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "x",
              int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        print("fps       :", round(cap.get(cv2.CAP_PROP_FPS), 2))
        print("frames    :", int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        cap.release()
    except ImportError:
        print("(install opencv-python to print video resolution / fps)")


def main():
    parser = argparse.ArgumentParser(description="Inspect a sign-language dataset")
    parser.add_argument("kind", choices=["islr", "include"])
    parser.add_argument("root", type=Path)
    args = parser.parse_args()

    if args.kind == "islr":
        inspect_islr(args.root)
    else:
        inspect_include(args.root)


if __name__ == "__main__":
    main()