"""Fix CSV files in evaluation/results/:
  - Fix column names: response -> model_output, query -> query_img
  - Rename files: {prefix}_result_{household}_{clothes}_{camera}.csv
               -> predictions_{prefix}_cropped_{clothes}_{household}_{camera}.csv
"""

from pathlib import Path

OLD_HEADER = "idx,label,prediction,response,query"
NEW_HEADER = "idx,label,prediction,model_output,query_img"

HOUSEHOLDS = {"family", "singleton"}
CLOTHES = {"crossclothes", "sameclothes"}
CAMERAS = {"crosscamera", "samecamera"}

results_dir = Path(__file__).parent / "results"
csv_files = list(results_dir.glob("*.csv"))
print(f"Found {len(csv_files)} CSV files\n")

header_fixed, header_skipped = 0, 0
renamed, rename_skipped = 0, 0

for path in sorted(csv_files):
    name = path.stem  # filename without .csv

    # --- Fix column header ---
    lines = path.read_text().splitlines(keepends=True)
    if lines and lines[0].strip() == OLD_HEADER:
        lines[0] = NEW_HEADER + "\n"
        path.write_text("".join(lines))
        header_fixed += 1
    else:
        header_skipped += 1

    # --- Rename file ---
    # Split on "_result_" to get prefix and suffix
    if "_result_" not in name:
        print(f"  Skipped rename (no '_result_'): {path.name}")
        rename_skipped += 1
        continue

    prefix, suffix = name.split("_result_", 1)
    # suffix = "{household}_{clothes}_{camera}"
    parts = suffix.split("_")
    # find household, clothes, camera by matching known values
    if len(parts) != 3 or parts[0] not in HOUSEHOLDS or parts[1] not in CLOTHES or parts[2] not in CAMERAS:
        print(f"  Skipped rename (unexpected suffix '{suffix}'): {path.name}")
        rename_skipped += 1
        continue

    household, clothes, camera = parts
    new_name = f"predictions_{prefix}_cropped_{clothes}_{household}_{camera}.csv"
    new_path = path.parent / new_name

    if new_path.exists():
        print(f"  Skipped rename (target exists): {new_name}")
        rename_skipped += 1
        continue

    path.rename(new_path)
    print(f"  Renamed: {path.name}")
    print(f"       -> {new_name}")
    renamed += 1

print(f"\nHeaders: {header_fixed} fixed, {header_skipped} skipped")
print(f"Renamed: {renamed} files, {rename_skipped} skipped")
