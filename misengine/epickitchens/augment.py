# Usage: python augment.py --split train --annotations_dir /path/to/epic-kitchens-100-annotations [--output train.xlsx]
#   Reads {annotations_dir}/EPIC_100_{split}.csv, writes {output} (default: {split}.xlsx)
import pandas as pd
import random 
import argparse

def get_Misaligned_Noun(group_df):
    group_df['Misaligned_Arg'] = [[] for _ in range(len(group_df))]

    for i, row in group_df.iterrows():
        rows_w_same_verb = group_df[(group_df.index != i) & (group_df["V"] == row['V'])]
        rows_misaligned_noun = rows_w_same_verb[rows_w_same_verb["ARG1"] != row["ARG1"]]

        indices_misaligned_noun = rows_misaligned_noun["Indices"].explode().dropna().tolist()

        group_df.at[i, "Misaligned_Arg"] = indices_misaligned_noun

    return group_df

def get_Misaligned_Verb(group_df):
    group_df['Misaligned_Verb'] = [[] for _ in range(len(group_df))]

    for i, row in group_df.iterrows():
        rows_w_same_noun = group_df[(group_df.index != i) & (group_df["ARG1"] == row["ARG1"])]
        rows_misaligned_verb = rows_w_same_noun[rows_w_same_noun["V"] != row["V"]]

        indices_misaligned_verb = rows_misaligned_verb["Indices"].explode().dropna().tolist()

        group_df.at[i, "Misaligned_Verb"] = indices_misaligned_verb

    return group_df

def get_Misaligned_Both(group_df):
    group_df['Misaligned_Both'] = [[] for _ in range(len(group_df))]

    for i, row in group_df.iterrows():
        rows_w_diff_noun = group_df[(group_df.index != i) & (group_df["ARG1"] != row["ARG1"])]
        rows_misaligned_verb = rows_w_diff_noun[rows_w_diff_noun["V"] != row["V"]]

        indices_misaligned_both = rows_misaligned_verb["Indices"].explode().dropna().tolist()

        group_df.at[i, "Misaligned_Both"] = indices_misaligned_both

    return group_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build misalignment-augmented split for EPIC-Kitchens-100.")
    parser.add_argument('--split', type=str, required=True, choices=['train', 'validation', 'test'], help='Which split to augment')
    parser.add_argument('--annotations_dir', type=str, required=True, help='Path to epic-kitchens-100-annotations directory')
    parser.add_argument('--output', type=str, default=None, help='Output xlsx (default: {split}.xlsx)')
    args = parser.parse_args()

    if args.output is None:
        args.output = f"{args.split}.xlsx"

    import os
    csv_path = os.path.join(args.annotations_dir, f"EPIC_100_{args.split}.csv")
    df = pd.read_csv(csv_path)
    df = df.rename(columns={'verb': 'V', 'noun': 'ARG1', 'stop_frame': 'end_frame'})

    df = df[
        (df["V"] != "None") & (df["V"] != "") &
        (df["ARG1"] != "None") & (df["ARG1"] != "")
    ]

    # Reset indices, so the impact of filtered-out rows is not felt
    df.reset_index(drop=True, inplace=True)

    group_df = df.groupby(['V', 'ARG1']).apply(lambda x: x.index.tolist()).reset_index()

    # Rename columns
    group_df.columns = ["V", "ARG1", "Indices"]

    # Ensures all indicies in the list are integers
    group_df['Indices'] = group_df['Indices'].apply(lambda x: [int(i) for i in x])

    # Retrieve the indices for the misaligned noun, verb, and both
    group_df = get_Misaligned_Noun(group_df)
    group_df = get_Misaligned_Verb(group_df)
    group_df = get_Misaligned_Both(group_df)

    full_df = {
        'video_id': [],
        'start_frame': [],
        'end_frame': [],
        'V': [],
        'ARG1': [],
        'label': [],
        'actual_V': [],
        'actual_ARG1': []
    }

    for i, row in group_df.iterrows():
        v = row["V"]
        arg = row['ARG1']

        aligned = row["Indices"]
        aligned_length = len(aligned)

        misaligned_v = row["Misaligned_Verb"]
        misaligned_arg = row["Misaligned_Arg"]
        misaligned_both = row["Misaligned_Both"]

        sampled_misaligned_v = random.sample(misaligned_v, min(len(misaligned_v), aligned_length))
        sampled_misaligned_arg = random.sample(misaligned_arg, min(len(misaligned_arg), aligned_length))
        sampled_misaligned_both = random.sample(misaligned_both, min(len(misaligned_both), aligned_length))

        for index in aligned:
            df_row = df.loc[index]

            full_df["video_id"].append(df_row["video_id"])
            full_df["start_frame"].append(df_row["start_frame"])
            full_df["end_frame"].append(df_row["end_frame"])
            full_df["V"].append(v)
            full_df["ARG1"].append(arg)
            full_df["label"].append(0)
            full_df["actual_V"].append(df_row["V"])
            full_df["actual_ARG1"].append(df_row["ARG1"])
        
        for index in sampled_misaligned_v:
            df_row = df.loc[index]

            full_df["video_id"].append(df_row["video_id"])
            full_df["start_frame"].append(df_row["start_frame"])
            full_df["end_frame"].append(df_row["end_frame"])
            full_df["V"].append(v)
            full_df["ARG1"].append(arg)
            full_df["label"].append(1)
            full_df["actual_V"].append(df_row["V"])
            full_df["actual_ARG1"].append(df_row["ARG1"])

        for index in sampled_misaligned_arg:
            df_row = df.loc[index]

            full_df["video_id"].append(df_row["video_id"])
            full_df["start_frame"].append(df_row["start_frame"])
            full_df["end_frame"].append(df_row["end_frame"])
            full_df["V"].append(v)
            full_df["ARG1"].append(arg)
            full_df["label"].append(2)
            full_df["actual_V"].append(df_row["V"])
            full_df["actual_ARG1"].append(df_row["ARG1"])

        for index in sampled_misaligned_both: 
            df_row = df.loc[index]

            full_df["video_id"].append(df_row["video_id"])
            full_df["start_frame"].append(df_row["start_frame"])
            full_df["end_frame"].append(df_row["end_frame"])
            full_df["V"].append(v)
            full_df["ARG1"].append(arg)
            full_df["label"].append(3)
            full_df["actual_V"].append(df_row["V"])
            full_df["actual_ARG1"].append(df_row["ARG1"])

    i_df = pd.DataFrame(full_df)

    label_counts = i_df["label"].value_counts().sort_index()
    label_percentages = i_df["label"].value_counts(normalize=True).sort_index() * 100

    print("Label Breakdown:")
    for label in label_counts.index:
        count = label_counts[label]
        percentage = label_percentages[label]
        print(f"Label {label}: {count} rows ({percentage:.2f}%)")

    i_df.to_excel(args.output, index=False)
