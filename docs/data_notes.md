# Data notes (Day 2)

Last updated: 2026-11-05

## 1. Shortlist

| Dataset | Language | Signs | Samples | Signers | Format | Download size | License / terms |
|---|---|---|---|---|---|---|---|
| Google ISLR (Kaggle `asl-signs`) | ASL | 250 | ~94.5k sequences | see inspection below | Landmarks as parquet (MediaPipe, 543 points/frame) | ~40 GB archive, ~57 GB extracted | Kaggle competition rules apply. **TODO: confirm license on the Kaggle data page** |
| WLASL | ASL | 2,000 glosses | 21,083 videos listed in the paper | 119 | Video links (YouTube etc.), many dead | Depends on how many links still work | C-UDA: academic and computational use only, no commercial use |
| ASL Citizen | ASL | 2,731 | 83,399 videos (~30 per sign) | 52 | Webcam videos | **TODO: check size on the download page** | Microsoft Research license (see project page); contact Microsoft for commercial use |
| INCLUDE (AI4Bharat) | ISL | 263 (15 categories) | 4,287 in paper, 4,292 on Zenodo | not stated in sources I checked | Videos (.MOV), split into zip parts | ~56.8 GB total on Zenodo | Reported as CC BY-NC-SA 4.0 (non-commercial). **TODO: confirm on the Zenodo record** |

Notes:
- Counts above come from papers and dataset pages, not from my own download. The sections below hold what I actually measured.
- WLASL: many glosses have only a few examples, and videos disappear over time, so the real count is lower than 21,083.
- INCLUDE: videos were recorded in Chennai, so it represents one regional variety of ISL.
- Non-commercial licenses (WLASL, INCLUDE) matter if this project is ever turned into a paid product.

## 2. Dataset I picked

**Choice:** Google ISLR 

**Why:** it’s already landmarks, well balanced, and 21 signers.

## 3. What I actually downloaded

- Download date: TODO
- Download method / command: TODO
- Disk used: TODO
- Files downloaded vs expected (for WLASL: how many videos actually worked): TODO

## 4. Inspection results

Run `python ml/inspect_dataset.py islr data/asl-signs` (or `include data/include`) and paste the output summary here.

- Format of one sample: parquet, one row per landmark per frame; columns frame, row_id, type, landmark_index, x, y, z; 543 landmarks/frame (468 face, 33 pose, 21 left hand, 21 right hand)
- Missing data: in one inspected clip (23 frames) face/pose 0% NaN, left hand 100%, right hand 52%; hands are often missing, so NaN handling is required
- Downloaded: train.csv + 1 sample landmark file (full archive ~40 GB not downloaded)



- Number of signs: TODO
- Total samples: TODO
- Samples per sign (min / median / max): TODO
- Number of signers: TODO
- Format of one sample (columns, frames, landmark types): TODO
- Missing data (fraction of NaN landmarks, e.g. hands not detected): TODO

## 5. Issues and decisions

split by participant_id, and fill missing hands with zeros plus a presence flag.
- TODO: anything surprising (unbalanced classes, one-hand recordings, missing files)
- TODO: what subset or split I will use for training

## 6. Next steps

- TODO
