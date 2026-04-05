# Usage: python augment.py --input clips.xlsx --clips_dir /path/to/ego4d/clips [--output all_clips_samples.xlsx]
import pandas as pd
import random 
import os
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
    parser = argparse.ArgumentParser(description="Build misalignment-augmented samples for Ego4D, filtering for available clips.")
    parser.add_argument('--input', type=str, default='clips.xlsx', help='Input xlsx from clips.py')
    parser.add_argument('--clips_dir', type=str, required=True, help='Path to directory containing Ego4D clip .mp4 files')
    parser.add_argument('--output', type=str, default='all_clips_samples.xlsx', help='Output xlsx')
    args = parser.parse_args()

    df = pd.read_excel(args.input)

    df = df[
        (df["V"] != "None") & (df["V"] != "") &
        (df["ARG1"] != "None") & (df["ARG1"] != "")
    ]
    df.reset_index(drop=True, inplace=True)

    Ego4D_list = os.listdir(args.clips_dir)
    valid_videos = set([video for video in Ego4D_list if video.endswith('.mp4')])

    def is_valid_clip1_id(clip_uid):
        return f"{clip_uid}.mp4" in valid_videos

    def is_valid_clip2_id(clip2_uid):
        return f"{clip2_uid}.mp4" in valid_videos or clip2_uid == 'Not required'

    df = df[df['clip1_uid'].apply(is_valid_clip1_id)]
    df = df[df['clip2_uid'].apply(is_valid_clip2_id)]

    group_df = df.groupby(['V', 'ARG1']).apply(lambda x: x.index.tolist()).reset_index()
    group_df.columns = ["V", "ARG1", "Indices"]
    group_df['Indices'] = group_df['Indices'].apply(lambda x: [int(i) for i in x])

    group_df = get_Misaligned_Noun(group_df)
    group_df = get_Misaligned_Verb(group_df)
    group_df = get_Misaligned_Both(group_df)

    full_df = {
        'clip1_uid': [],
        'clip1_start_frame': [],
        'clip1_end_frame': [],
        'clip2_uid': [],
        'clip2_start_frame': [],
        'clip2_end_frame': [],
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

            full_df["clip1_uid"].append(df_row["clip1_uid"])
            full_df["clip1_start_frame"].append(df_row["clip1_start_frame"])
            full_df["clip1_end_frame"].append(df_row["clip1_end_frame"])
            full_df["clip2_uid"].append(df_row["clip2_uid"])
            full_df["clip2_start_frame"].append(df_row["clip2_start_frame"])
            full_df["clip2_end_frame"].append(df_row["clip2_end_frame"])        
            full_df["V"].append(v)
            full_df["ARG1"].append(arg)
            full_df["label"].append(0)
            full_df["actual_V"].append(df_row["V"])
            full_df["actual_ARG1"].append(df_row["ARG1"])

        for index in sampled_misaligned_v:
            df_row = df.loc[index]

            full_df["clip1_uid"].append(df_row["clip1_uid"])
            full_df["clip1_start_frame"].append(df_row["clip1_start_frame"])
            full_df["clip1_end_frame"].append(df_row["clip1_end_frame"])
            full_df["clip2_uid"].append(df_row["clip2_uid"])
            full_df["clip2_start_frame"].append(df_row["clip2_start_frame"])
            full_df["clip2_end_frame"].append(df_row["clip2_end_frame"])        
            full_df["V"].append(v)
            full_df["ARG1"].append(arg)
            full_df["label"].append(1)
            full_df["actual_V"].append(df_row["V"])
            full_df["actual_ARG1"].append(df_row["ARG1"])

        for index in sampled_misaligned_arg:
            df_row = df.loc[index]

            full_df["clip1_uid"].append(df_row["clip1_uid"])
            full_df["clip1_start_frame"].append(df_row["clip1_start_frame"])
            full_df["clip1_end_frame"].append(df_row["clip1_end_frame"])
            full_df["clip2_uid"].append(df_row["clip2_uid"])
            full_df["clip2_start_frame"].append(df_row["clip2_start_frame"])
            full_df["clip2_end_frame"].append(df_row["clip2_end_frame"])        
            full_df["V"].append(v)
            full_df["ARG1"].append(arg)
            full_df["label"].append(2)
            full_df["actual_V"].append(df_row["V"])
            full_df["actual_ARG1"].append(df_row["ARG1"])

        for index in sampled_misaligned_both: 
            df_row = df.loc[index]

            full_df["clip1_uid"].append(df_row["clip1_uid"])
            full_df["clip1_start_frame"].append(df_row["clip1_start_frame"])
            full_df["clip1_end_frame"].append(df_row["clip1_end_frame"])
            full_df["clip2_uid"].append(df_row["clip2_uid"])
            full_df["clip2_start_frame"].append(df_row["clip2_start_frame"])
            full_df["clip2_end_frame"].append(df_row["clip2_end_frame"])        
            full_df["V"].append(v)
            full_df["ARG1"].append(arg)
            full_df["label"].append(3)
            full_df["actual_V"].append(df_row["V"])
            full_df["actual_ARG1"].append(df_row["ARG1"])

    df = pd.DataFrame(full_df)

    label_counts = df["label"].value_counts().sort_index()
    label_percentages = df["label"].value_counts(normalize=True).sort_index() * 100

    print("Label Breakdown:")
    for label in label_counts.index:
        count = label_counts[label]
        percentage = label_percentages[label]
        print(f"Label {label}: {count} rows ({percentage:.2f}%)")

    df.to_excel(args.output, index=False)
