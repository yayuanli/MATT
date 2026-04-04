# Usage: python annotation.py --json_path /path/to/annotation.json [--output annotation.xlsx]
import json
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="Parse EgoPER annotation.json into a flat xlsx with decoded action/action_type strings.")
parser.add_argument('--json_path', type=str, required=True, help='Path to EgoPER annotation.json')
parser.add_argument('--output', type=str, default='annotation.xlsx', help='Output xlsx path')
args = parser.parse_args()

with open(args.json_path, 'r') as f:
    data = json.load(f)

rows = []

for name in ['coffee', 'pinwheels', 'oatmeal', 'quesadilla', 'tea']:
    segments = data.get(name, None)
    segments = segments.get('segments', None)
    for segment in segments:
        video_id = segment.get('video_id', None)

        labels = segment.get('labels', None)
        actions = labels.get('action', [])
        actiontypes = labels.get('action_type', [])
        timestamps = labels.get('time_stamp', [])
        errordescriptions = labels.get('error_description', [])
        
        for action, action_type, time_stamp, error_description in zip(actions, actiontypes, timestamps, errordescriptions):
            rows.append({
                'video_id': video_id,
                'action': action,
                'action_type': action_type,
                'timestamp': time_stamp,
                'error_description': error_description
            })

df = pd.DataFrame(rows)

start_row = 0
for name in ['coffee', 'pinwheels', 'oatmeal', 'quesadilla', 'tea']:
    segments = data.get(name, None)
    action2idx = segments.get('action2idx', None)
    actiontype2idx = segments.get('actiontype2idx', None)

    action2idx = {v: k for k, v in action2idx.items()}
    actiontype2idx = {v: k for k, v in actiontype2idx.items()}

    count = 0

    for idx, row in df.iloc[start_row:].iterrows():
        video_id = row['video_id']
        if video_id.split('_')[0] == name:
            if df.at[idx, 'action'] in action2idx:
                df.at[idx, 'action'] = action2idx[df.at[idx, 'action']]
            
            if df.at[idx, 'action_type'] in actiontype2idx:
                df.at[idx, 'action_type'] = actiontype2idx[df.at[idx, 'action_type']]
        else:
            break

        count += 1
    start_row += count
    
df.to_excel(args.output, index=False)
