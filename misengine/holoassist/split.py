# Usage: python split.py --input df_fg_output.xlsx --train_ids /path/to/train.txt --val_ids /path/to/val.txt --test_ids /path/to/test.txt [--output_dir .]
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="Split HoloAssist annotations into train/validation/test by video ID lists.")
parser.add_argument('--input', type=str, default='df_fg_output.xlsx', help='Input xlsx from df_fg.py')
parser.add_argument('--train_ids', type=str, required=True, help='Path to train video ID list (.txt)')
parser.add_argument('--val_ids', type=str, required=True, help='Path to validation video ID list (.txt)')
parser.add_argument('--test_ids', type=str, required=True, help='Path to test video ID list (.txt)')
parser.add_argument('--output_dir', type=str, default='.', help='Output directory for split xlsx files')
args = parser.parse_args()

df_output = pd.read_excel(args.input)
print(f"Total rows: {len(df_output)}")

def load_ids(filename):
    with open(filename, "r") as f:
        # strip whitespace and ignore empty lines
        return set(line.strip() for line in f if line.strip())

train_ids = load_ids(args.train_ids)
val_ids   = load_ids(args.val_ids)
test_ids  = load_ids(args.test_ids)

# Check that there is no overlap between these sets
overlap_train_val = train_ids.intersection(val_ids)
overlap_train_test = train_ids.intersection(test_ids)
overlap_val_test   = val_ids.intersection(test_ids)
if overlap_train_val or overlap_train_test or overlap_val_test:
    print("Error: Overlapping video IDs detected!")
    if overlap_train_val:
        print("Overlap between train and validation:", overlap_train_val)
    if overlap_train_test:
        print("Overlap between train and test:", overlap_train_test)
    if overlap_val_test:
        print("Overlap between validation and test:", overlap_val_test)
    exit(1)
else:
    print("No overlaps found between train, validation, and test IDs.")

# Partition df_output into three DataFrames based on video_id.
# (Assuming your DataFrame has a column "video_id". Convert to string for safety.)
df_train = df_output[df_output["video_id"].astype(str).isin(train_ids)]
df_val   = df_output[df_output["video_id"].astype(str).isin(val_ids)]
df_test  = df_output[df_output["video_id"].astype(str).isin(test_ids)]

import os
os.makedirs(args.output_dir, exist_ok=True)

df_train.to_excel(os.path.join(args.output_dir, "train_base.xlsx"), index=False)
df_val.to_excel(os.path.join(args.output_dir, "validation_base.xlsx"), index=False)
df_test.to_excel(os.path.join(args.output_dir, "test_base.xlsx"), index=False)

print(f"Files saved: train_base.xlsx ({len(df_train)} rows), validation_base.xlsx ({len(df_val)} rows), test_base.xlsx ({len(df_test)} rows)")
