# Usage: python create_splits.py --input all_clips_samples.xlsx [--output_dir .] [--seed 42]
import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser(description="Split Ego4D augmented samples into train/valid/test (80/10/10 random split).")
parser.add_argument('--input', type=str, required=True, help='Input xlsx (e.g. all_clips_samples.xlsx from augment.py)')
parser.add_argument('--output_dir', type=str, default='.', help='Directory to save split xlsx files')
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducible shuffle')
args = parser.parse_args()

df = pd.read_excel(args.input)
num_rows = len(df)

df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)

df_train = df.iloc[:int(num_rows * 0.8)]
df_valid = df.iloc[int(num_rows * 0.8):int(num_rows * 0.8 + num_rows * 0.1)]
df_test = df.iloc[int(num_rows * 0.8 + num_rows * 0.1):]

os.makedirs(args.output_dir, exist_ok=True)

df_train.to_excel(os.path.join(args.output_dir, 'train.xlsx'), index=False)
df_valid.to_excel(os.path.join(args.output_dir, 'valid.xlsx'), index=False)
df_test.to_excel(os.path.join(args.output_dir, 'test.xlsx'), index=False)

print(f"Train: {len(df_train)} rows, Valid: {len(df_valid)} rows, Test: {len(df_test)} rows")
print(f"Files saved to {args.output_dir}/")
