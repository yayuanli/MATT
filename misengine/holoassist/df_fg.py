# Usage: python df_fg.py --json_path /path/to/data-annotation-trainval-v1_1.json [--output df_fg_output.xlsx]
import json
import pandas as pd
import math 
import argparse

parser = argparse.ArgumentParser(description="Parse HoloAssist fine-grained action annotations from JSON to xlsx, filtering ambiguous segments.")
parser.add_argument('--json_path', type=str, required=True, help='Path to HoloAssist annotation JSON')
parser.add_argument('--output', type=str, default='df_fg_output.xlsx', help='Output xlsx path')
args = parser.parse_args()

with open(args.json_path, "r") as f:
    data = json.load(f)

rows = []

# Process each record in the JSON file
for record in data:
    # Get the video name from the top level (video_name)
    video_name = record.get("video_name", "")
    task_id = record.get("taskId", "")
    task_type = record.get("taskType", "")
    fps = int(math.ceil((record.get("videoMetadata", {}).get("video", {}).get("fps", ""))))
    
    # Iterate through the events in the record
    for event in record.get("events", []):
        if event.get("label") == "Fine grained action":
            event_id = event.get("id")
            start = event.get("start")
            end = event.get("end")
            attributes = event.get("attributes", {})
            
            # FOR FINE GRAINED ONLY!

            correctness = attributes.get("Action Correctness", "")
            explanation = attributes.get("Incorrect Action Explanation", "")
            by = attributes.get("Incorrect Action Corrected by", "")

            # Extracting key attributes if they exist
            action_sentence = attributes.get("Action sentence", "")
            verb = attributes.get("Verb", "")
            adjective = attributes.get("Adjective", "")
            noun = attributes.get("Noun", "")

            start_frame = int(math.floor(float(start) * float(fps)))
            end_frame = int(math.floor(float(end) * float(fps)))
            
            # Create a dictionary for this row
            row = {
                "video_id": video_name,
                "taskId": task_id,
                "taskType": task_type,
                "event_id": event_id,
                "fps": fps,
                "start_time": start,
                "end_time": end,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "narration": action_sentence,
                "V": verb,
                "ARG1": noun,
                "Adjective": adjective,
                "Action Correctness": correctness,
                "Incorrect Action Explanation": explanation,
                "Incorrect Action Corrected by": by
            }
            rows.append(row)

df = pd.DataFrame(rows)

# Filter out ambiguous segments: correct-action rows whose (video_id, start_frame, end_frame)
# group has more than one unique V or ARG1
three_cols = ["video_id", "start_frame", "end_frame"]
correct = df[df["Action Correctness"] == "Correct Action"]
ambig_segments = (
    correct.groupby(three_cols)
           .filter(lambda g: g["V"].nunique() > 1 or g["ARG1"].nunique() > 1)
           [three_cols]
           .drop_duplicates()
)

if len(ambig_segments) > 0:
    df = (
        df.merge(ambig_segments.assign(_ambig=1), on=three_cols, how="left")
          .query("_ambig.isna()")
          .drop(columns="_ambig")
          .reset_index(drop=True)
    )
    print(f"Removed {len(ambig_segments)} ambiguous segments")

df.to_excel(args.output, index=False)
print(f"Excel file saved as {args.output}")
