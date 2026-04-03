# Usage: python extract_frames.py --input annotation.xlsx --egoper_dir /path/to/EgoPER --output_dir /path/to/frames
import subprocess
import os
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="Extract video frames for all EgoPER videos at 15 fps using ffmpeg.")
parser.add_argument('--input', type=str, default='annotation.xlsx', help='Input annotation xlsx')
parser.add_argument('--egoper_dir', type=str, required=True, help='EgoPER data directory containing video files')
parser.add_argument('--output_dir', type=str, required=True, help='Output directory for extracted frames')
args = parser.parse_args()

df = pd.read_excel(args.input)
video_ids = df['video_id'].unique().tolist()

os.makedirs(args.output_dir, exist_ok=True)

for video_id in video_ids:
    
    food = video_id.split('_', 1)[0]

    if food == "coffee" or food == "oatmeal":
        video_id_path = os.path.join(args.egoper_dir, "trim_videos", video_id + ".mp4")
    else: 
        video_id_path = os.path.join(args.egoper_dir, food, "trim_videos", video_id + ".mp4")

    frames_path = os.path.join(args.output_dir, str(video_id) + "_frames")
    os.makedirs(frames_path, exist_ok=True)

    frame_path_pattern = os.path.join(frames_path, '%06d.png')
    ffmpeg_cmd = ['ffmpeg', '-i', video_id_path, '-vf', "fps=15, scale=640:360", '-vsync', 'vfr', frame_path_pattern]
    try:
        subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL, check=True)
        print(f"Frames extracted for clip: {video_id_path}")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred for video: {video_id_path} - FFmpeg said: {e.stderr}")
