"""Fix CSV column names: query_id -> idx, answer -> prediction."""

from pathlib import Path

OLD_HEADER = "query_id,label,answer,model_output"
NEW_HEADER = "idx,label,prediction,model_output"

runs_dir = Path(__file__).parent / "runs"
csv_files = list(runs_dir.rglob("*.csv"))
print(f"Found {len(csv_files)} CSV files")

fixed, skipped = 0, 0
for path in csv_files:
    lines = path.read_text().splitlines(keepends=True)
    if lines and lines[0].strip() == OLD_HEADER:
        lines[0] = NEW_HEADER + "\n"
        path.write_text("".join(lines))
        print(f"  Fixed: {path.relative_to(runs_dir)}")
        fixed += 1
    else:
        print(f"  Skipped (unexpected header): {path.relative_to(runs_dir)}")
        skipped += 1

print(f"\nDone: {fixed} fixed, {skipped} skipped")
