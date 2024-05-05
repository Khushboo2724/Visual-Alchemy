import cv2
import numpy as np

# Define the output video file path and format
output_video_path = 'output_video-bicupicEquiRectangular.mp4'

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


import cv2
import numpy as np

def bicubic_interpolation(input_frame, target_width, target_height):
    # Resize the input frame using bicubic interpolation
    resized_frame = cv2.resize(input_frame, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
    return resized_frame

def convert_to_equirectangular(input_frame, width, height):
    # Initialize an empty equirectangular frame
    equirectangular_frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Define spherical coordinates conversion
    for y in range(height):
        for x in range(width):
            # Convert pixel coordinates (x, y) to spherical coordinates
            theta = (x / width) * 2 * np.pi  # azimuth (longitude)
            phi = (y / height) * np.pi  # inclination (latitude)
            
            # Convert spherical coordinates back to pixel coordinates in the input frame
            src_x = int((theta / (2 * np.pi)) * input_frame.shape[1])
            src_y = int((phi / np.pi) * input_frame.shape[0])
            
            # Set pixel value from input frame to the equirectangular frame
            equirectangular_frame[y, x] = input_frame[src_y, src_x]
    
    return equirectangular_frame

def process_video_to_equirectangular(input_video_path, output_video_path):
    # Open the input video
    cap = cv2.VideoCapture(input_video_path)
    
    # Get video properties
    input_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    input_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Define equirectangular video width and height
    width = 2 * input_width  # Equirectangular width is twice the input width
    height = input_height   # Equirectangular height is equal to input height

    # Create video writer for the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # Process each frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Apply bicubic interpolation to the frame
        frame_resized = bicubic_interpolation(frame, input_width, input_height)

        # Convert the frame to equirectangular format
        equirectangular_frame = convert_to_equirectangular(frame_resized, width, height)
        
        # Write the equirectangular frame to the output video
        out.write(equirectangular_frame)
    
    # Release video capture and writer
    cap.release()
    out.release()
    print(f"Equirectangular video saved to {output_video_path}")

# Example usage
input_video_path = "C:/Users/DELL/Downloads/4053041-hd_1280_720_50fps (1) (1) (1) (1) (1).mp4"  # Specify the input video file path
output_video_path = "C:/Users/DELL/Downloads/output_video-bicupicEquiRectangular.mp4" # Specify the output video file path

# Process the video to create an equirectangular video
process_video_to_equirectangular(input_video_path, output_video_path)





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
video_path1 = "C:/Users/DELL/Downloads/4053041-hd_1280_720_50fps (1) (1) (1) (1) (1).mp4" #give path to input org file
video_path2 =  "C:/Users/DELL/Downloads/output_video-bicupicEquiRectangular.mp4" # Specify the output video file path

# Call the function to continuously play the videos side by side
play_videos_side_by_side_continuous(video_path1, video_path2, fixed_width=320)
