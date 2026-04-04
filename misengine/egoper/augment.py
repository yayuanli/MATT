# Usage: python augment.py --input_dir /path/to/EgoPER_processing [--output all/train.xlsx]
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
    parser = argparse.ArgumentParser(description="Build misalignment-augmented training set for EgoPER (all food categories combined).")
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing per-food training.xlsx files (e.g. coffee/training.xlsx)')
    parser.add_argument('--output', type=str, default='all/train.xlsx', help='Output xlsx path (default: all/train.xlsx)')
    args = parser.parse_args()

    foods = ["coffee", "oatmeal", "tea", "pinwheels", "quesadilla"]
    file_paths = [os.path.join(args.input_dir, food, "training.xlsx") for food in foods]

    dfs = []
    for fp in file_paths:
        df1 = pd.read_excel(fp)
        dfs.append(df1)
    df = pd.concat(dfs, ignore_index=True)

    df = df[
        (df["V"] != "None") & (df["V"] != "") &
        (df["ARG1"] != "None") & (df["ARG1"] != "") &
        (df["Error_V"] != "None") & (df["Error_V"] != "") &
        (df["Error_Arg1"] != "None") & (df["Error_Arg1"] != "")
    ]

    errors_found = (df["action_type"] != "Normal").any()
    print(f"Error samples found in training: {errors_found}")

    errors_df = df[df["action_type"] != 'Normal']

    df = df[df["action_type"] == 'Normal']
    df.reset_index(drop=True, inplace=True)

    group_df = df.groupby(['V', 'ARG1']).apply(lambda x: x.index.tolist()).reset_index()
    group_df.columns = ["V", "ARG1", "Indices"]
    group_df['Indices'] = group_df['Indices'].apply(lambda x: [int(i) for i in x])

    group_df = get_Misaligned_Noun(group_df)
    group_df = get_Misaligned_Verb(group_df)
    group_df = get_Misaligned_Both(group_df)

    full_df = {
        'video_id': [],
        'start_frame': [],
        'end_frame': [],
        'V': [],
        'Arg1': [],
        'label': [],
        'actual_V': [],
        'actual_Arg1': []
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
            full_df["Arg1"].append(arg)
            full_df["label"].append(0)
            full_df["actual_V"].append(df_row["V"])
            full_df["actual_Arg1"].append(df_row["ARG1"])
        
        for index in sampled_misaligned_v:
            df_row = df.loc[index]

            full_df["video_id"].append(df_row["video_id"])
            full_df["start_frame"].append(df_row["start_frame"])
            full_df["end_frame"].append(df_row["end_frame"])
            full_df["V"].append(v)
            full_df["Arg1"].append(arg)
            full_df["label"].append(1)
            full_df["actual_V"].append(df_row["V"])
            full_df["actual_Arg1"].append(df_row["ARG1"])

        for index in sampled_misaligned_arg:
            df_row = df.loc[index]

            full_df["video_id"].append(df_row["video_id"])
            full_df["start_frame"].append(df_row["start_frame"])
            full_df["end_frame"].append(df_row["end_frame"])
            full_df["V"].append(v)
            full_df["Arg1"].append(arg)
            full_df["label"].append(2)
            full_df["actual_V"].append(df_row["V"])
            full_df["actual_Arg1"].append(df_row["ARG1"])

        for index in sampled_misaligned_both: 
            df_row = df.loc[index]

            full_df["video_id"].append(df_row["video_id"])
            full_df["start_frame"].append(df_row["start_frame"])
            full_df["end_frame"].append(df_row["end_frame"])
            full_df["V"].append(v)
            full_df["Arg1"].append(arg)
            full_df["label"].append(3)
            full_df["actual_V"].append(df_row["V"])
            full_df["actual_Arg1"].append(df_row["ARG1"])

    final_df = pd.DataFrame(full_df)
    label_counts = final_df["label"].value_counts().sort_index()
    label_percentages = final_df["label"].value_counts(normalize=True).sort_index() * 100

    print("Label Breakdown:")
    for label in label_counts.index:
        count = label_counts[label]
        percentage = label_percentages[label]
        print(f"Label {label}: {count} rows ({percentage:.2f}%)")

    # Alternate rows by cycling through label groups
    groups = [group for _, group in final_df.groupby('label')]
    result = []
    while any(not group.empty for group in groups):
        for group in groups:
            if not group.empty:
                result.append(group.iloc[0])
                group.drop(group.index[0], inplace=True)

    alt_final_df = pd.DataFrame(result).reset_index(drop=True)

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    alt_final_df.to_excel(args.output)

    final_df_set = set(map(tuple, final_df.to_numpy()))
    alt_final_df_set = set(map(tuple, alt_final_df.to_numpy()))

    if final_df_set == alt_final_df_set:
        print("Both DataFrames contain the same rows.")
    else:
        print("The DataFrames do not contain the same rows.")
