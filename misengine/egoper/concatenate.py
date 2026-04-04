# Usage: python concatenate.py --input_dir /path/to/EgoPER_processing [--output_dir all]
import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser(description="Concatenate per-food validation and test splits into combined xlsx files.")
parser.add_argument('--input_dir', type=str, required=True, help='Directory containing per-food split xlsx files (e.g. coffee/validation.xlsx)')
parser.add_argument('--output_dir', type=str, default='all', help='Output directory for combined splits (default: all)')
args = parser.parse_args()

foods = ["coffee", "oatmeal", "tea", "pinwheels", "quesadilla"]
os.makedirs(args.output_dir, exist_ok=True)

for split in ["validation", "test"]:
    dfs = []
    for food in foods:
        fp = os.path.join(args.input_dir, food, f"{split}.xlsx")
        df = pd.read_excel(fp)
        print(f"{food}/{split}: {len(df)} rows")
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    out_path = os.path.join(args.output_dir, f"{split}.xlsx")
    combined.to_excel(out_path, index=False)
    print(f"Combined {split}: {len(combined)} rows -> {out_path}")
