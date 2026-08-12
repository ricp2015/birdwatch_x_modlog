from pathlib import Path
import pandas as pd
import csv

INPUT_DIR = Path("data/interim/reddit")
OUTPUT_DIR = INPUT_DIR / "csv"

def convert():
    """Convert convert to the target format."""
    parquet_files = list(INPUT_DIR.rglob("*.parquet"))
    if not parquet_files:
        print("No parquet files found.")
        return
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for parquet_path in parquet_files:
        if "csv" in parquet_path.parts:
            continue
        relative_path = parquet_path.relative_to(INPUT_DIR)
        csv_path = (OUTPUT_DIR / relative_path).with_suffix(".csv")
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Converting: {relative_path}")
        try:
            df = pd.read_parquet(parquet_path)
            df.to_csv(
                csv_path,
                index=False,
                quoting=csv.QUOTE_ALL,
                escapechar="\\"
            )
        except Exception as e:
            print(f"[WARNING] Failed: {parquet_path} ({e})")
    print(f"\nCSVs saved in: {OUTPUT_DIR}")

if __name__ == "__main__":
    convert()
