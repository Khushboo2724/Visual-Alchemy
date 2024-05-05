import cv2
import numpy as np

# Define the output video file path and format
output_video_path = 'output_video.mp4'

# Define the codec and create VideoWriter object
# Use 'mp4v' codec for MP4 format
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 30  # Frames per second
width, height = 640, 480  # Frame dimensions (resolution)

# Create a VideoWriter object
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# Define the number of frames to create (e.g., 1 second of video at 30 fps)
num_frames = fps  # 1 second of video

# Create a black frame using NumPy
black_frame = np.zeros((height, width, 3), dtype=np.uint8)

# Write the black frame repeatedly to create an empty video file
for _ in range(num_frames):
    out.write(black_frame)

# Release the VideoWriter object
out.release()
print(f'Empty video file created: {output_video_path}')



# Import necessary libraries
import cv2
import numpy as np
from IPython.display import Video  # Import Video for displaying video in notebook

# Function to compute the depth map
def compute_depth_map(frame):
    # Apply depth estimation algorithm to the input frame
    # Demonstrate a simple depth estimation method
    disparity_map = np.ones_like(frame, dtype=np.float32) * 50
    return disparity_map

# Function to apply depth-based effects
def depth_based_effects(frame, disparity_map):
    # Apply depth-based effects such as depth-of-field blur
    blurred_frame = cv2.GaussianBlur(frame, (15, 15), 0)
    depth_effect_frame = cv2.addWeighted(frame, 0.7, blurred_frame, 0.3, 0)
    return depth_effect_frame

# Main function
def main():
    # Open video capture object
    video_path="C:/Users/DELL/Downloads/4053041-hd_1280_720_50fps.mp4"
    video_capture = cv2.VideoCapture(video_path)

    # Get video properties (width, height, fps)
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))

    # Define output video path
    output_video_path = "C:/Users/DELL/Downloads/output_video.mp4"

    # Create video writer object for output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Process video frames
    while True:
        # Read a frame from the video
        ret, frame = video_capture.read()
        if not ret:
            break  # End of video

        # Compute depth map from the input frame
        disparity_map = compute_depth_map(frame)

        # Apply depth-based effects
        depth_effect_frame = depth_based_effects(frame, disparity_map)

        # Write processed frame to output video
        output_video.write(depth_effect_frame)

    # Release video capture and writer objects
    video_capture.release()
    output_video.release()

    print(f'Processed video saved as: {output_video_path}')

    # Display the final video file in the notebook
    return Video(output_video_path)



def play_videos_side_by_side_continuous(video_path1, video_path2, fixed_width=320):
    # Open video capture objects for both videos
    cap1 = cv2.VideoCapture(video_path1)
    cap2 = cv2.VideoCapture(video_path2)

    # Check if both video captures are opened successfully
    if not cap1.isOpened() or not cap2.isOpened():
        print("Failed to open one or both video files.")
        return

    while True:
        # Read frames from both videos
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        # If either video ends, restart both videos
        if not ret1 or not ret2:
            cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
            cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # Get the current dimensions of the frames
        height1, width1 = frame1.shape[:2]
        height2, width2 = frame2.shape[:2]

        # Calculate the scale ratio based on the fixed width
        scale1 = fixed_width / width1
        scale2 = fixed_width / width2

        # Resize frames to the fixed width while maintaining the aspect ratio
        frame1_resized = cv2.resize(frame1, (fixed_width, int(height1 * scale1)))
        frame2_resized = cv2.resize(frame2, (fixed_width, int(height2 * scale2)))

        # Concatenate frames horizontally
        combined_frame = cv2.hconcat([frame1_resized, frame2_resized])

        # Display the concatenated frame
        cv2.imshow('Side by Side Videos', combined_frame)

        # Break the loop if the user presses the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture objects and close all OpenCV windows
    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()

# Set the video file paths
video_path1 = "C:/Users/DELL/OneDrive/Documents/multimedia project/fastMotion-blur - Made with Clipchamp (1).mp4" #give path to input org file
video_path2 = "C:/Users/DELL/OneDrive/Documents/multimedia project/output_videoBlurNEW.mp4" #give path to the new generated video file


# Call the function to continuously play the videos side by side
play_videos_side_by_side_continuous(video_path1, video_path2, fixed_width=320)

