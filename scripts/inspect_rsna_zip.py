from pathlib import Path
from collections import Counter
import zipfile

ZIP_PATH = Path("data/raw/rsna/rsna-intracranial-hemorrhage-detection.zip")
OUT_DIR = Path("data/raw/rsna")
REPORT_PATH = Path("data/raw/rsna/zip_file_list_preview.txt")

if not ZIP_PATH.exists():
    raise FileNotFoundError(f"ZIP not found: {ZIP_PATH}")

print("===== RSNA ZIP INSPECTION =====")
print("ZIP:", ZIP_PATH)
print("Size GB:", round(ZIP_PATH.stat().st_size / (1024 ** 3), 2))

with zipfile.ZipFile(ZIP_PATH, "r") as z:
    names = z.namelist()

    print("Total files inside ZIP:", len(names))

    print("\n===== FIRST 80 FILES =====")
    for name in names[:80]:
        print(name)

    suffix_counter = Counter(Path(name).suffix.lower() for name in names)
    print("\n===== FILE EXTENSIONS =====")
    for suffix, count in suffix_counter.most_common(20):
        print(f"{suffix or '[no suffix]'}: {count}")

    csv_files = [name for name in names if name.lower().endswith(".csv")]
    zip_files = [name for name in names if name.lower().endswith(".zip")]

    print("\n===== CSV FILES =====")
    for name in csv_files:
        print(name)

    print("\n===== NESTED ZIP FILES =====")
    for name in zip_files:
        print(name)

    REPORT_PATH.write_text("\n".join(names[:5000]), encoding="utf-8")
    print("\nPreview file list saved to:", REPORT_PATH)

    print("\n===== EXTRACT CSV FILES ONLY =====")
    for name in csv_files:
        print("Extracting:", name)
        z.extract(name, OUT_DIR)

print("\nSTATUS: ZIP INSPECTION DONE")
