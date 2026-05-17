from pathlib import Path
import zipfile

import pandas as pd


ZIP_PATH = Path("data/raw/rsna/rsna-intracranial-hemorrhage-detection.zip")
WIDE_CSV = Path("data/processed/rsna/stage_2_train_wide.csv")

SAMPLE_DIR = Path("data/sample/rsna_dicom")
MANIFEST_PATH = Path("data/sample/rsna_sample_manifest.csv")

ZIP_PREFIX = "rsna-intracranial-hemorrhage-detection/stage_2_train"

N_NORMAL = 50
N_POSITIVE = 50
RANDOM_STATE = 42


def main() -> None:
    if not ZIP_PATH.exists():
        raise FileNotFoundError(f"ZIP not found: {ZIP_PATH}")

    if not WIDE_CSV.exists():
        raise FileNotFoundError(f"Wide CSV not found: {WIDE_CSV}")

    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    print("===== LOAD WIDE LABEL CSV =====")
    df = pd.read_csv(WIDE_CSV)

    print("Wide shape:", df.shape)
    print("Columns:", df.columns.tolist())

    normal_df = df[df["any"] == 0].sample(
        n=N_NORMAL,
        random_state=RANDOM_STATE,
    )

    positive_df = df[df["any"] == 1].sample(
        n=N_POSITIVE,
        random_state=RANDOM_STATE,
    )

    sample_df = pd.concat([normal_df, positive_df], axis=0)
    sample_df = sample_df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

    sample_df["zip_member"] = sample_df["image_id"].apply(
        lambda image_id: f"{ZIP_PREFIX}/{image_id}.dcm"
    )

    sample_df["local_path"] = sample_df["image_id"].apply(
        lambda image_id: str(SAMPLE_DIR / f"{image_id}.dcm")
    )

    print("\n===== SAMPLE LABEL DISTRIBUTION =====")
    print(sample_df[["any", "epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural"]].sum())

    print("\n===== OPEN ZIP AND EXTRACT SAMPLE DICOM =====")
    extracted = 0
    missing = []

    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        zip_names = set(z.namelist())

        for _, row in sample_df.iterrows():
            member = row["zip_member"]
            image_id = row["image_id"]
            output_path = SAMPLE_DIR / f"{image_id}.dcm"

            if member not in zip_names:
                missing.append(member)
                continue

            if output_path.exists() and output_path.stat().st_size > 0:
                extracted += 1
                continue

            with z.open(member) as source, open(output_path, "wb") as target:
                target.write(source.read())

            extracted += 1

    sample_df.to_csv(MANIFEST_PATH, index=False)

    print("Extracted/existing DICOM:", extracted)
    print("Missing DICOM:", len(missing))

    if missing:
        print("First missing items:")
        for item in missing[:10]:
            print(item)

    print("\nSaved manifest:", MANIFEST_PATH)
    print("Sample DICOM folder:", SAMPLE_DIR)

    print("\nSTATUS: SAMPLE DICOM EXTRACTION DONE")


if __name__ == "__main__":
    main()
