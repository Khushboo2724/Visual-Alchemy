#!/usr/bin/env python
# coding: utf-8

# In[4]:


import cv2
import numpy as np

# Define the output video file path and format
output_video_path = 'output_video-Anaglyph.mp4'

# Define the codec and create VideoWriter object
# Use 'mp4v' codec for MP4 format
fourcc = cv2.VideoWriter_fourcc(*'h264')
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


# In[6]:


import cv2
import numpy as np


def estimate_depth(frame):
    # Placeholder depth estimation function.
    # In practice, use a more sophisticated algorithm like a neural network for depth estimation.
    # Returns a constant disparity map for demonstration purposes.
    disparity_map = np.ones_like(frame[:, :, 0], dtype=np.float32) * 50
    return disparity_map

def generate_anaglyph(frame, depth_map, shift_scale=0.1):
    # Generate left and right views using depth map
    rows, cols, _ = frame.shape
    left_view = np.zeros_like(frame)
    right_view = np.zeros_like(frame)
    
    # Shift each pixel based on the depth map
    for y in range(rows):
        for x in range(cols):
            shift = int(depth_map[y, x] * shift_scale)
            
            # Left view shifted to the left
            if x + shift < cols:
                left_view[y, x] = frame[y, x + shift]
            
            # Right view shifted to the right
            if x - shift >= 0:
                right_view[y, x] = frame[y, x - shift]
    
    # Create anaglyph frame using red-cyan filter
    anaglyph_frame = np.zeros_like(frame)
    
    # Red channel from left view
    anaglyph_frame[:, :, 2] = left_view[:, :, 2]  # Red channel
    
    # Green and blue channels from right view
    anaglyph_frame[:, :, :2] = right_view[:, :, :2]  # Cyan channel
    
    return anaglyph_frame

def process_anaglyph_video(input_video_path, output_video_path):
    # Open video capture object
    video_capture = cv2.VideoCapture(input_video_path)
    
    # Get video properties
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    
    # Create video writer for the output anaglyph video
    fourcc = cv2.VideoWriter_fourcc(*'h264')
    output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    while True:
        # Read frame from video
        ret, frame = video_capture.read()
        if not ret:
            break
        
        # Estimate depth map from the input frame
        depth_map = estimate_depth(frame)
        
        # Generate anaglyph frame using the estimated depth and the input frame
        anaglyph_frame = generate_anaglyph(frame, depth_map)
        
        # Write the anaglyph frame to the output video
        output_video.write(anaglyph_frame)
    
    # Release video capture and writer objects
    video_capture.release()
    output_video.release()
    print(f"Anaglyph video saved to {output_video_path}")

# Example usage
input_video_path = "C:/Users/DELL/Downloads/4053041-hd_1280_720_50fps (1) (1) (1) (1) (1).mp4"  # Path to the input video
output_video_path = "C:/Users/DELL/Downloads/output_video-Anaglyph.mp4"  # Path to the output anaglyph video

# Process the input video and generate an anaglyph video
process_anaglyph_video(input_video_path, output_video_path)




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
video_path1 = "C:/Users/DELL/Downloads/4053041-hd_1280_720_50fps (1) (1) (1) (1) (1).mp4" #give path toinput org file 
video_path2 = "C:/Users/DELL/Downloads/output_video-Anaglyph.mp4" #give path to the new generated video file

# Call the function to continuously play the videos side by side
play_videos_side_by_side_continuous(video_path1, video_path2, fixed_width=320)



# In[ ]:




