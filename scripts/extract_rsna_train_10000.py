from pathlib import Path
import zipfile

import pandas as pd


ZIP_PATH = Path("data/raw/rsna/rsna-intracranial-hemorrhage-detection.zip")
WIDE_CSV = Path("data/processed/rsna/stage_2_train_wide.csv")
HOLDOUT_MANIFEST = Path("data/processed/rsna_holdout_5000_manifest.csv")

TRAIN_DIR = Path("data/processed/rsna_train_10000_dicom")
TRAIN_MANIFEST = Path("data/processed/rsna_train_10000_manifest.csv")

ZIP_PREFIX = "rsna-intracranial-hemorrhage-detection/stage_2_train"

N_NORMAL = 5000
N_POSITIVE = 5000
RANDOM_STATE = 2026

LABEL_COLUMNS = [
    "any",
    "epidural",
    "intraparenchymal",
    "intraventricular",
    "subarachnoid",
    "subdural",
]


def main() -> None:
    if not ZIP_PATH.exists():
        raise FileNotFoundError(f"ZIP not found: {ZIP_PATH}")

    if not WIDE_CSV.exists():
        raise FileNotFoundError(f"Wide CSV not found: {WIDE_CSV}")

    if not HOLDOUT_MANIFEST.exists():
        raise FileNotFoundError(f"Holdout manifest not found: {HOLDOUT_MANIFEST}")

    TRAIN_DIR.mkdir(parents=True, exist_ok=True)

    print("===== LOAD FULL LABEL CSV =====")
    df = pd.read_csv(WIDE_CSV)
    holdout_df = pd.read_csv(HOLDOUT_MANIFEST)

    holdout_ids = set(holdout_df["image_id"].tolist())

    candidate_df = df[~df["image_id"].isin(holdout_ids)].copy()

    print("Full dataset shape:", df.shape)
    print("Holdout IDs excluded:", len(holdout_ids))
    print("Candidate train pool:", candidate_df.shape)

    normal_pool = candidate_df[candidate_df["any"] == 0]
    positive_pool = candidate_df[candidate_df["any"] == 1]

    print("Normal pool:", len(normal_pool))
    print("Positive pool:", len(positive_pool))

    normal_df = normal_pool.sample(
        n=N_NORMAL,
        random_state=RANDOM_STATE,
    )

    positive_df = positive_pool.sample(
        n=N_POSITIVE,
        random_state=RANDOM_STATE,
    )

    train_df = pd.concat([normal_df, positive_df], axis=0)
    train_df = train_df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

    train_df["zip_member"] = train_df["image_id"].apply(
        lambda image_id: f"{ZIP_PREFIX}/{image_id}.dcm"
    )

    train_df["local_path"] = train_df["image_id"].apply(
        lambda image_id: str(TRAIN_DIR / f"{image_id}.dcm")
    )

    print("\n===== TRAIN 10000 LABEL DISTRIBUTION =====")
    print(train_df[LABEL_COLUMNS].sum())

    print("\nTrain any value counts:")
    print(train_df["any"].value_counts().sort_index())

    print("\n===== EXTRACT TRAIN 10000 DICOM FILES =====")

    extracted = 0
    missing = []

    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        for idx, row in train_df.iterrows():
            member = row["zip_member"]
            image_id = row["image_id"]
            output_path = TRAIN_DIR / f"{image_id}.dcm"

            if output_path.exists() and output_path.stat().st_size > 0:
                extracted += 1
                continue

            try:
                with z.open(member) as source, open(output_path, "wb") as target:
                    target.write(source.read())
                extracted += 1
            except KeyError:
                missing.append(member)

            if (idx + 1) % 500 == 0:
                print(f"Processed {idx + 1}/{len(train_df)} | extracted/existing={extracted} | missing={len(missing)}")

    train_df.to_csv(TRAIN_MANIFEST, index=False)

    print("\nExtracted/existing DICOM:", extracted)
    print("Missing DICOM:", len(missing))
    print("Saved train manifest:", TRAIN_MANIFEST)
    print("Train DICOM dir:", TRAIN_DIR)

    if missing:
        print("First missing:")
        for item in missing[:10]:
            print(item)

    print("\nSTATUS: TRAIN 10000 EXTRACTION DONE")


if __name__ == "__main__":
    main()
