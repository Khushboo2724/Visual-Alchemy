# Visual-Alchemy
Enhancing Monocular Videos with Depth-Based Effects and  Anaglyph Conversion

**Project Description:**
VisualAlchemy is a cutting-edge project that focuses on transforming standard monocular videos into spatial videos enriched with depth-based effects like parallax, depth-based lighting, blur, and anaglyph conversion. This project aims to provide users with an immersive and engaging viewing experience by leveraging advanced computer vision techniques and innovative visual effects.

**Key Features:**
1. Depth Estimation: Utilize depth estimation algorithms to extract depth information from monocular videos.
2. Spatial Video Generation: Convert 2D videos into spatial videos by incorporating depth information and re-projecting them into a 3D space.
3. Depth-Based Effects: Implement parallax, depth-based lighting, blur effects to enhance the spatial videos.
4. Anaglyph Conversion: Convert spatial videos into anaglyph format for stereoscopic 3D viewing.
5. 360-Degree View Transformation: Transform spatial videos for 360-degree viewing, providing an interactive experience.

**Methodology:**
1. Input Data Acquisition: Obtain 2D videos from various sources.
2. Depth Estimation: Utilize computer vision techniques for depth estimation.
3. Spatial Video Generation: Convert 2D videos into spatial videos by incorporating depth information.
4. Depth-Based Effects: Implement parallax, depth-based lighting, blur effects.
5. Anaglyph Video Conversion: Convert spatial videos into anaglyph format for 3D viewing.
6. Monocular Video Processing: All steps are performed on monocular videos captured with a single camera.
7. The videos used for processing are highly compressed in order for fastly processing

**Usage:**
Clone the repository to your local machine.
Install the necessary dependencies.
Run the project to start transforming monocular videos into immersive spatial experiences.
You can either run these effects individually or we have created a streamlit app where you can choose the effect you want to apply.
The processing of these vids takes a considerable time to generate.

**Some Tips on which type of video would this effect would be comprehensible more clearly-**
1. For Blurr effect(Depth of Field)- Fast paced videos
2. For Depth-Depenedent Lighting-  Underwater videos
3. For Parallax effect- Layered videos or videos shot with multi-cameras
4. For Anaglyph effect- any above generated video on applying effect can be converted into anaglyph to see the effects applied to the videos in a 3D view by using the anaglyph glasses(glasses with red and blue 
   filters on either eye frame)
5. For 360 degree- Just as much any video will give you the effect
(I have given a sample vid in the code repo)
I AM EVEN BETTER GIVING ALL THE OUPUT VIDEO FILES GENERATED IN THE REPO

**Enjoy transforming your videos into captivating spatial experiences with VisualAlchemy!**
