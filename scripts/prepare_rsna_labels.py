from pathlib import Path

import pandas as pd


RAW_DIR = Path("data/raw/rsna/rsna-intracranial-hemorrhage-detection")
PROCESSED_DIR = Path("data/processed/rsna")

TRAIN_CSV = RAW_DIR / "stage_2_train.csv"
SUBMISSION_CSV = RAW_DIR / "stage_2_sample_submission.csv"

OUT_WIDE_CSV = PROCESSED_DIR / "stage_2_train_wide.csv"
OUT_DISTRIBUTION_CSV = PROCESSED_DIR / "label_distribution.csv"
OUT_DUPLICATE_CSV = PROCESSED_DIR / "duplicate_label_groups.csv"
OUT_REPORT_TXT = PROCESSED_DIR / "label_report.txt"

LABEL_COLUMNS = [
    "any",
    "epidural",
    "intraparenchymal",
    "intraventricular",
    "subarachnoid",
    "subdural",
]


def parse_rsna_id(value: str) -> tuple[str, str]:
    """
    RSNA label ID format example:
    ID_000012eaf_epidural

    Converts to:
    image_id = ID_000012eaf
    subtype = epidural
    """
    parts = value.split("_")
    if len(parts) < 3:
        raise ValueError(f"Unexpected ID format: {value}")

    image_id = "_".join(parts[:2])
    subtype = "_".join(parts[2:])
    return image_id, subtype


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if not TRAIN_CSV.exists():
        raise FileNotFoundError(f"Train CSV not found: {TRAIN_CSV}")

    print("===== LOAD TRAIN CSV =====")
    train_df = pd.read_csv(TRAIN_CSV)

    print("Train CSV path:", TRAIN_CSV)
    print("Train shape:", train_df.shape)
    print("\nTrain head:")
    print(train_df.head(12))

    print("\nTrain columns:")
    print(train_df.columns.tolist())

    if "ID" not in train_df.columns or "Label" not in train_df.columns:
        raise ValueError("Expected columns ['ID', 'Label'] not found.")

    print("\n===== PARSE ID =====")
    parsed = train_df["ID"].apply(parse_rsna_id)
    train_df["image_id"] = parsed.apply(lambda x: x[0])
    train_df["subtype"] = parsed.apply(lambda x: x[1])

    unique_subtypes = sorted(train_df["subtype"].unique().tolist())
    print("Unique subtypes:")
    print(unique_subtypes)

    unknown_subtypes = sorted(set(unique_subtypes) - set(LABEL_COLUMNS))
    if unknown_subtypes:
        raise ValueError(f"Unknown subtypes found: {unknown_subtypes}")

    print("\n===== CHECK ROWS PER IMAGE =====")
    rows_per_image = train_df.groupby("image_id").size()
    print(rows_per_image.describe())

    duplicate_image_groups = rows_per_image[rows_per_image != 6]
    print("Image groups not equal to 6 rows:", len(duplicate_image_groups))

    if len(duplicate_image_groups) > 0:
        duplicate_detail = (
            train_df[train_df["image_id"].isin(duplicate_image_groups.index)]
            .sort_values(["image_id", "subtype", "ID"])
        )
        duplicate_detail.to_csv(OUT_DUPLICATE_CSV, index=False)
        print("Duplicate label detail saved:", OUT_DUPLICATE_CSV)
        print(duplicate_image_groups.head(20))
        print("Action: continue using pivot_table aggfunc='max' to safely merge duplicate binary labels.")

    print("\n===== PIVOT LONG TO WIDE =====")
    wide_df = (
        train_df
        .pivot_table(
            index="image_id",
            columns="subtype",
            values="Label",
            aggfunc="max",
        )
        .reset_index()
    )

    wide_df.columns.name = None

    for col in LABEL_COLUMNS:
        if col not in wide_df.columns:
            wide_df[col] = 0

    wide_df = wide_df[["image_id"] + LABEL_COLUMNS]
    wide_df[LABEL_COLUMNS] = wide_df[LABEL_COLUMNS].astype(int)

    wide_df["dcm_filename"] = wide_df["image_id"] + ".dcm"

    subtype_cols = [col for col in LABEL_COLUMNS if col != "any"]
    wide_df["num_positive_subtypes"] = wide_df[subtype_cols].sum(axis=1)

    print("Wide shape:", wide_df.shape)
    print("\nWide head:")
    print(wide_df.head())

    print("\n===== LABEL CONSISTENCY CHECK =====")
    mismatch_any = wide_df[
        ((wide_df["num_positive_subtypes"] > 0) & (wide_df["any"] == 0)) |
        ((wide_df["num_positive_subtypes"] == 0) & (wide_df["any"] == 1))
    ]
    print("Any-label mismatch count:", len(mismatch_any))

    print("\n===== LABEL DISTRIBUTION =====")
    distribution = []

    total_images = len(wide_df)

    for col in LABEL_COLUMNS:
        positive = int(wide_df[col].sum())
        negative = int(total_images - positive)
        prevalence = positive / total_images if total_images > 0 else 0.0

        distribution.append({
            "label": col,
            "positive": positive,
            "negative": negative,
            "prevalence": prevalence,
        })

    dist_df = pd.DataFrame(distribution)
    print(dist_df)

    print("\n===== MULTI-LABEL SUMMARY =====")
    multi_summary = wide_df["num_positive_subtypes"].value_counts().sort_index()
    print(multi_summary)

    print("\n===== SAVE OUTPUTS =====")
    wide_df.to_csv(OUT_WIDE_CSV, index=False)
    dist_df.to_csv(OUT_DISTRIBUTION_CSV, index=False)

    report_lines = []
    report_lines.append("RSNA Intracranial Hemorrhage Label Report")
    report_lines.append("=" * 60)
    report_lines.append(f"Raw train shape: {train_df.shape}")
    report_lines.append(f"Wide train shape: {wide_df.shape}")
    report_lines.append(f"Total unique images: {total_images}")
    report_lines.append(f"Image groups not equal to 6 rows: {len(duplicate_image_groups)}")
    report_lines.append(f"Any-label mismatch count: {len(mismatch_any)}")
    report_lines.append("")
    report_lines.append("Label distribution:")
    report_lines.append(dist_df.to_string(index=False))
    report_lines.append("")
    report_lines.append("Positive subtype count per image:")
    report_lines.append(multi_summary.to_string())

    OUT_REPORT_TXT.write_text("\n".join(report_lines), encoding="utf-8")

    print("Saved:", OUT_WIDE_CSV)
    print("Saved:", OUT_DISTRIBUTION_CSV)
    print("Saved:", OUT_REPORT_TXT)

    if SUBMISSION_CSV.exists():
        submission_df = pd.read_csv(SUBMISSION_CSV)
        print("\n===== SAMPLE SUBMISSION =====")
        print("Submission shape:", submission_df.shape)
        print(submission_df.head())

    print("\nSTATUS: LABEL PREPARATION DONE")


if __name__ == "__main__":
    main()
