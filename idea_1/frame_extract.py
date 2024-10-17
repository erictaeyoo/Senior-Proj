import cv2
import os
import sys
import glob
import random

def extract_frames_from_video(video_path, output_folder, frame_rate=1, video_id=0):
    """
    Extract frames from a video file and save them as images in the output folder.

    Parameters:
    - video_path: Path to the input video file.
    - output_folder: Folder to save the extracted frames.
    - frame_rate: Extract one frame every 'frame_rate' frames.
    - video_id: Unique identifier for the video to avoid filename collisions.
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created directory: {output_folder}")

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames in video: {frame_count}")

    current_frame = 0
    saved_frame = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # No more frames to read

        # Save every 'frame_rate' frames
        if current_frame % frame_rate == 0:
            # Use video_id and saved_frame to generate unique filenames for each frame
            frame_filename = os.path.join(output_folder, f"video_{video_id}_frame_{saved_frame:05d}_real.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_frame += 1

        current_frame += 1

    cap.release()
    print(f"Extracted {saved_frame} frames from {video_path}")


# For getting frames from multiple videos
video_directory = sys.argv[1]  # Path to the directory containing your video files
output_directory = sys.argv[2]  # Path to the output directory
num_videos_to_extract = int(sys.argv[3])  # Number of videos to extract frames from

# Supported video formats
video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv']

# Get a list of all video files in the directory
video_files = []
for ext in video_extensions:
    video_files.extend(glob.glob(os.path.join(video_directory, ext)))

print(f"Found {len(video_files)} video files.")

# Check if the requested number of videos is more than available videos
if num_videos_to_extract > len(video_files):
    num_videos_to_extract = len(video_files)
    print(f"Requested number of videos exceeds available videos. Adjusting to {num_videos_to_extract}.")

# Randomly select a subset of videos
selected_videos = random.sample(video_files, num_videos_to_extract)

# Extract frames from each selected video
for video_id, video_file in enumerate(selected_videos):
    # Extract frames and store them in the shared output folder with unique filenames
    extract_frames_from_video(video_file, output_directory, frame_rate=1, video_id=video_id)
