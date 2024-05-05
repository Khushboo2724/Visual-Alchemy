import streamlit as st
from streamlit_lottie import st_lottie
import cv2
from IPython.display import Video
from streamlit_lottie import st_lottie
import requests
#from Anaglyph_integrated import Anaglyph

# Function to load and display pre-processed videos
def display_processed_video(video_path, uploaded_file):
    #st.video(video_path)
    
    # Display original and processed videos side by side
    st.header("Original vs Processed Video")
    col1, col2 = st.columns(2)
    col1.video(uploaded_file)
    col2.video(video_path)

# Streamlit app
def main():
    st.title("VISUAL ALCHEMY")
    st.markdown(
        """
        <style>
        .reportview-container {
            background: linear-gradient(to right, #4b0082, #7b68ee);
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    def load_lottieurl(url):
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()   

    #loading assets 
    lottie_coding = load_lottieurl("https://lottie.host/ca5574f7-b2e3-4b2f-ad52-6b0a894d60bc/JPEUEyVMwg.json")    

    right_column = st.sidebar

    with right_column:
      st_lottie(lottie_coding, height=500, key="coding")

    # Display multimedia processing information
    st.header("About Video Processing")
    st.write("Video processing treats a video like a flip book - a series of still images. Each image (frame) is analyzed and adjusted. This can involve editing like cutting or adding effects, compressing the video for storage or streaming, or even using AI to understand what's happening in the video itself! ")
    st.write("This project allows users to experiment with depth information to create unique visual enhancements on their uploaded videos. ")
  
    # File uploader for video input
    st.header("Upload Video")
    uploaded_file = st.file_uploader("Choose an MP4 video file", type="mp4")

    st.write("Choose an algorithm and upload a video to see the processing result.")

    if uploaded_file is not None:
        # Display uploaded video
        st.video(uploaded_file)

        # Processing options
        st.header("Choose Processing Algorithm")
        effect_type = st.selectbox(
            "Select an effect",
            ["None", "Depth of Field", "Parallax Method", "Depth Dependent Lighting", "360 Degree View", "Anaglyph Video"],
        )

        if st.button("Process"):
            if effect_type == "Depth of Field":
                # Display pre-processed Depth of Field video
                '''anaglyph=Anaglyph(uploaded_file)
                output_final=anaglyph.main()'''
                st.header("Depth of Field i.e Blur Effect is seen more prominently in Fast Motion Videos") 
                display_processed_video("C:/Users/DELL/OneDrive/Documents/multimedia project/output_videoBlurNEW.mp4", uploaded_file)

            elif effect_type == "Parallax Method":
                # Display pre-processed Parallax Method video
                st.header("Parallax Method is seen more prominently in videos shot with Moving Camera of Videos that are Layered - Intensity 10, shitf scale- 0.1")
                display_processed_video("C:/Users/DELL/OneDrive/Documents/multimedia project/output_video-2parallaxNEW1.mp4", uploaded_file)

            elif effect_type== "Depth Dependent Lighting":
                st.header("Depth Dependent Lighting is more prominently seen in Underwater videos- Intensity 100")
                display_processed_video("C:/Users/DELL/OneDrive/Documents/multimedia project/output_videoLightingNEW1.mp4", uploaded_file)

            elif effect_type=="360 Degree View":
                #display_processed_video("C:/Users/DELL/Downloads/Telegram Desktop/output_video-bicupicEquiRectangular2.mp4", uploaded_file)
                st.video("C:/Users/DELL/Downloads/Telegram Desktop/output_video-bicupicEquiRectangular2.mp4")

            elif effect_type=="Anaglyph Video":
                #display_processed_video("C:/Users/DELL/Downloads/output_video-Anaglyph.mp4", uploaded_file)
                st.video("C:/Users/DELL/Downloads/output_video-Anaglyph.mp4")


            else:
                
                st.header("No processing option selected. \n Displaying original video")
                st.video(uploaded_file)
                         

if __name__ == '__main__':
    main()
