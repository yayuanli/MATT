import subprocess
import os

clips_dir = "/scratch1/home/yayuanli/dat/Ego4D/v2/v2/clips"
output_directory = "/scratch1/home/aaditj/clips_frames"

clips = [f.replace('.mp4', '') for f in os.listdir(clips_dir) if f.endswith('.mp4')]

for clip in clips:
    clip_path = os.path.join(clips_dir, clip + ".mp4")
    frame_path = os.path.join(output_directory, clip + "_frames")
    os.makedirs(frame_path, exist_ok=True)
    frame_path_pattern = os.path.join(frame_path, '%05d.png')

    ffmpeg_cmd = ['ffmpeg', '-i', clip_path, '-vf', 'scale=640:360', '-vsync', 'vfr', frame_path_pattern]

    try:
        subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
        print(f"Frames extracted for clip: {clip}")
    except subprocess.CalledProcessError as e:
        print(f"Error for {clip}: {e.stderr.decode() if e.stderr else 'unknown error'}")