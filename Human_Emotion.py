

import streamlit as st
import numpy as np
import cv2
from deepface import DeepFace
from PIL import Image
import tempfile 
 

st.title("Human Emotion Recognation")

st.write("Uploaded an Image or Video")

options=st.selectbox('Choose an Option...',('Image','Video'))

def Analyze_Emotion(image_or_video):
    try:
        analysis=DeepFace.analyze(image_or_video,actions=['emotion'],enforce_detection=False)
        return analysis[0]['emotion']

    except ValueError as e:
        st.write(f'Error : (e)')
        return None
        
if options=='Image':
    upload=st.file_uploader('Please Upload An Image',type=['png','jpg','jpeg','wepd'])

    if upload is not None:
        img=Image.open(upload)
        img_array=np.array(img)
        st.image(img_array, channels = 'RGB')

        emotion_scores=Analyze_Emotion(img_array)

        if emotion_scores:
            detected_emotion=max(emotion_scores,key=emotion_scores.get)
            st.write(f'Detected Emotion:{detected_emotion}')

        else:
            st.write('No Face Deteced in Your Image')

if options=='Video':
    upload = st.file_uploader('Please, Upload a Video...', type = ['mp4', 'mov', 'avi'])

    if upload is not None:
        with tempfile.NamedTemporaryFile(delete = False) as temp_video:
            temp_video.write(upload.read())
            video_path = temp_video.name
        video = cv2.VideoCapture(video_path)

        frame_rate = 40 #control how many frames
        frame_count = 0

        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            frame_count += 1

            #process and diaply frames everytime
            if frame_count % frame_rate == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                emotion_scores = Analyze_Emotion(frame_rgb)
                if emotion_scores:
                    detected_emotion = max(emotion_scores, key = emotion_scores.get)
                    cv2.putText(frame, detected_emotion, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,255), 2)
                else:
                    detected_emotion = 'No Face Detected'
                    cv2.putText(frame, detected_emotion, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,255), 2)

                st.image(frame, channels = 'BGR')
        video.release()
            
