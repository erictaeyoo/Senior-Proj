import cv2
import os
import sys

def extract_frames_from_video(video_path, output_folder, frame_rate=1):
    """
    Extract frames from a video file and save them as images.

    Parameters:
    - video_path: Path to the input video file.
    - output_folder: Folder to save the extracted frames.
    - frame_rate: Extract one frame every 'frame_rate' frames.
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
            frame_filename = os.path.join(output_folder, f"frame_{saved_frame:05d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_frame += 1

        current_frame += 1

    cap.release()
    print(f"Extracted {saved_frame} frames from {video_path}")

extract_frames_from_video(sys.argv[1], sys.argv[2])

# For getting frames from multiple videos
# import glob

# # Path to the directory containing your video files
# video_directory = '/path/to/video/files'  # Update this path
# output_directory = '/path/to/output/frames'  # Update this path

# # Supported video formats
# video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv']

# # Get a list of all video files in the directory
# video_files = []
# for ext in video_extensions:
#     video_files.extend(glob.glob(os.path.join(video_directory, ext)))

# print(f"Found {len(video_files)} video files.")

# # Extract frames from each video
# for video_file in video_files:
#     # Get the base name of the video file (without extension)
#     video_name = os.path.splitext(os.path.basename(video_file))[0]
    
#     # Create a specific output folder for each video
#     video_output_folder = os.path.join(output_directory, video_name)
    
#     # Extract frames
#     extract_frames_from_video(video_file, video_output_folder, frame_rate=1)  # Adjust 'frame_rate' as needed
