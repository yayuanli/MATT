# Usage: python clips.py --metadata /path/to/ego4d.json --input parquet.xlsx --output clips.xlsx
import json 
from collections import defaultdict
import pandas as pd
import sys 
import argparse

parser = argparse.ArgumentParser(description="Map Ego4D video-level frame ranges to clip-level coordinates.")
parser.add_argument('--metadata', type=str, required=True, help='Path to ego4d.json metadata file')
parser.add_argument('--input', type=str, default='parquet.xlsx', help='Input xlsx (cleaned parquet export)')
parser.add_argument('--output', type=str, default='clips.xlsx', help='Output xlsx with clip mappings')
args = parser.parse_args()

with open(args.metadata, 'r') as metadata:
    metadata = json.load(metadata)

clips = metadata.get('clips')

if(len(clips) == 0):
    print("ERROR: DATA NOT PROPERLY LOADED")
    sys.exit()

lookup_table = defaultdict(list)

for clip in clips:
    video_uid = clip['video_uid']
    start_frame = int(clip['video_start_frame'])
    end_frame = int(clip['video_end_frame']) # End frame is exclusive
    clip_uid = clip['clip_uid']
    lookup_table[video_uid].append((start_frame, end_frame, clip_uid))

# Sort the clips in for each video. Sort occurs based on the first element of the tuples, 
# then later ones if tie-breakers are needed
for video_uid in lookup_table:
    lookup_table[video_uid].sort()

df = pd.read_excel(args.input)
df[['start_frame', 'end_frame']] = df[['start_frame', 'end_frame']].astype(int)

'''
NOTE: Not needed if the provided parquet.xlsx has already been filtered.
Uncomment if starting from a raw parquet export.

keys = ["video_uid", "start_frame", "end_frame"]
ambiguous_keys = (
    df.groupby(keys)
      .filter(lambda g: g['V'].nunique() > 1 or g['ARG1'].nunique() > 1)
      [keys]
      .drop_duplicates()
)
if len(ambiguous_keys) > 0:
    df = (
        df.merge(ambiguous_keys.assign(_ambig=1), on=keys, how="left")
          .query("_ambig.isna()")
          .drop(columns="_ambig")
          .reset_index(drop=True)
    )
    print(f"Removed {len(ambiguous_keys)} ambiguous segments ({len(df)} rows remain)")

# Remove the mialignment columns because the row numbers no longer correspond to the same entry
df = df.drop(columns=["MisalignSRL_V", "MisalignSRL_ARG1", "MisalignSRL_V_ARG1"], errors="ignore")
'''

df['clip1_uid'] = 'Not found'
df['clip1_start_frame'] = -1
df['clip1_end_frame'] = -1

df['clip2_uid'] = 'Not required'
df['clip2_start_frame'] = -1
df['clip2_end_frame'] = -1

for index, row in df.iterrows():
    video_uid = row['video_uid']
    start_frame = row['start_frame'] # Start frame of the video
    end_frame = row['end_frame'] #End frame of the video

    clips = lookup_table[video_uid] # Clips of that video

    for i in range(len(clips)):
        if(int(clips[i][0]) <= int(start_frame)): # if the start frame of the clip is less than or equal to the start frame of the moment (This means the clip could contain the moment)
            if(int(clips[i][1]) >= (int(end_frame))): # if the end frame of the clip is greater than the end frame of the moment (i.e. if one clip encapsulates the entire moment). Note that end frame is exclusive
                df.at[index, 'clip1_uid'] = str(clips[i][2])
                df.at[index, 'clip1_start_frame'] = (int(start_frame) - int(clips[i][0])) # Start frame is in the larger video. 
                df.at[index, 'clip1_end_frame'] = (int(end_frame) - int(clips[i][0])) # Remains exclusive

                # The end frame of the clip is exclusive 
                break
            
            elif(int(clips[i][1]) <= int(end_frame) and int(clips[i][1]) > int(start_frame)): # if the end frame of the clip is less than the end frame of the moment (i.e. the clip ends earlier),
                # and the end frame of the clip is greater than the starting clip of the moment
                df.at[index, 'clip1_uid'] = str(clips[i][2])
                df.at[index, 'clip1_start_frame'] = (int(start_frame) - int(clips[i][0]))
                df.at[index, 'clip1_end_frame'] = (int(clips[i][1]) - int(clips[i][0])) # End frame of the clip minus the start of the clip
                
                total = int(int(end_frame) - int(start_frame))
                total_sofar = int(int(clips[i][1]) - int(start_frame))

                i+=1 # The next clip should finish the previous one

                if(i < len(clips)): 
                    if(clips[i][0] < end_frame): # if the starting frame of the clip is less than the end frame of the moment
                        df.at[index, 'clip2_uid'] = str(clips[i][2])
                        df.at[index, 'clip2_start_frame'] = int(0)
                        df.at[index, 'clip2_end_frame'] = int(total - total_sofar)
                    else:
                        df.at[index, 'clip2_uid'] = str("Required, but not found")
                else:
                    df.at[index, 'clip2_uid'] = str("Required, but not found")

                break

df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])

df.to_excel(args.output, index=False)
print(f"DataFrame with added columns has been saved to {args.output}")
