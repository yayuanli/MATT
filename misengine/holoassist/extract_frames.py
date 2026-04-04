# Usage: python extract_frames.py --input df_fg_output.xlsx --video_base_path /path/to/HoloAssist/video_pitch_shifted [--ffmpeg_path ffmpeg]
import os
import pandas as pd
import subprocess
import argparse

parser = argparse.ArgumentParser(description="Extract video frames for HoloAssist videos using ffmpeg.")
parser.add_argument('--input', type=str, default='df_fg_output.xlsx', help='Input xlsx from df_fg.py')
parser.add_argument('--video_base_path', type=str, required=True, help='Base path containing HoloAssist video directories')
parser.add_argument('--ffmpeg_path', type=str, default='ffmpeg', help='Path to ffmpeg binary (default: ffmpeg)')
args = parser.parse_args()

df = pd.read_excel(args.input, engine="openpyxl")

print("Starting frame extraction")
for subdir in os.listdir(args.video_base_path):
    subdir_path = os.path.join(args.video_base_path, subdir)
    video_frames_path = os.path.join(subdir_path, "Export_py", "video_frames")
    
    # Check if it's a directory.
    if os.path.isdir(subdir_path) and not os.path.exists(video_frames_path):
        video_id = subdir
        
        # Look up the first matching row in the DataFrame.
        match = df[df["video_id"] == video_id]
        if match.empty:
            print(f"Video ID '{video_id}' not found in excel file. Skipping.")
            continue
        
        target_fps = match.iloc[0]["fps"]
        print(f"\nProcessing video '{video_id}' with target fps: {target_fps}")

        # Build the path to the Export_py folder and the video file.
        export_py_path = os.path.join(subdir_path, "Export_py")
        video_file = os.path.join(export_py_path, "Video_pitchshift.mp4")
        
        if not os.path.isfile(video_file):
            print(f"Video file not found: {video_file}. Skipping.")
            continue
        
        # Create the output directory for frames.
        frames_output_dir = os.path.join(export_py_path, "video_frames")
        os.makedirs(frames_output_dir, exist_ok=True)
        
        # Build the output file pattern.
        output_pattern = os.path.join(frames_output_dir, "frame_%05d.jpg")
        
        # Build the ffmpeg command.
        # -y: overwrite output files without asking
        # -i: input file
        # -vf fps=<target_fps>: use the fps video filter to sample frames at target_fps
        ffmpeg_command = [
            args.ffmpeg_path, "-y",
            "-i", video_file,
            "-vf", f"fps={target_fps}",
            output_pattern
        ]
        
        print("Running ffmpeg command:", " ".join(ffmpeg_command))
        try:
            subprocess.run(ffmpeg_command, check=True)
            print(f"Frames extracted for video '{video_id}' at target fps {target_fps}.")
        except subprocess.CalledProcessError as e:
            print(f"ffmpeg failed for video '{video_id}': {e}")
