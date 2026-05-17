from pathlib import Path
import zipfile

import pandas as pd


ZIP_PATH = Path("data/raw/rsna/rsna-intracranial-hemorrhage-detection.zip")
WIDE_CSV = Path("data/processed/rsna/stage_2_train_wide.csv")
TRAIN_2000_MANIFEST = Path("data/processed/rsna_2000_manifest.csv")

HOLDOUT_DIR = Path("data/processed/rsna_holdout_5000_dicom")
HOLDOUT_MANIFEST = Path("data/processed/rsna_holdout_5000_manifest.csv")

ZIP_PREFIX = "rsna-intracranial-hemorrhage-detection/stage_2_train"

N_HOLDOUT = 5000
RANDOM_STATE = 777

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

    if not TRAIN_2000_MANIFEST.exists():
        raise FileNotFoundError(f"Training manifest not found: {TRAIN_2000_MANIFEST}")

    HOLDOUT_DIR.mkdir(parents=True, exist_ok=True)

    print("===== LOAD FULL LABEL CSV =====")
    df = pd.read_csv(WIDE_CSV)
    used_df = pd.read_csv(TRAIN_2000_MANIFEST)

    used_ids = set(used_df["image_id"].tolist())

    candidate_df = df[~df["image_id"].isin(used_ids)].copy()

    print("Full dataset shape:", df.shape)
    print("Used training/sample IDs:", len(used_ids))
    print("Candidate holdout pool:", candidate_df.shape)

    holdout_df = candidate_df.sample(
        n=N_HOLDOUT,
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)

    holdout_df["zip_member"] = holdout_df["image_id"].apply(
        lambda image_id: f"{ZIP_PREFIX}/{image_id}.dcm"
    )

    holdout_df["local_path"] = holdout_df["image_id"].apply(
        lambda image_id: str(HOLDOUT_DIR / f"{image_id}.dcm")
    )

    print("\n===== HOLDOUT LABEL DISTRIBUTION =====")
    print(holdout_df[LABEL_COLUMNS].sum())

    print("\nHoldout any value counts:")
    print(holdout_df["any"].value_counts().sort_index())

    prevalence = holdout_df["any"].mean()
    print("Holdout any prevalence:", prevalence)

    print("\n===== EXTRACT HOLDOUT DICOM FILES =====")
    extracted = 0
    missing = []

    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        for idx, row in holdout_df.iterrows():
            member = row["zip_member"]
            image_id = row["image_id"]
            output_path = HOLDOUT_DIR / f"{image_id}.dcm"

            if output_path.exists() and output_path.stat().st_size > 0:
                extracted += 1
                continue

            try:
                with z.open(member) as source, open(output_path, "wb") as target:
                    target.write(source.read())
                extracted += 1
            except KeyError:
                missing.append(member)

            if (idx + 1) % 250 == 0:
                print(f"Processed {idx + 1}/{len(holdout_df)} | extracted/existing={extracted} | missing={len(missing)}")

    holdout_df.to_csv(HOLDOUT_MANIFEST, index=False)

    print("\nExtracted/existing DICOM:", extracted)
    print("Missing DICOM:", len(missing))
    print("Saved holdout manifest:", HOLDOUT_MANIFEST)
    print("Holdout DICOM dir:", HOLDOUT_DIR)

    if missing:
        print("First missing:")
        for item in missing[:10]:
            print(item)

    print("\nSTATUS: NATURAL HOLDOUT EXTRACTION DONE")


if __name__ == "__main__":
    main()
