import cv2
import mediapipe as medp
import numpy as np
import streamlit as st

# initialize mediapipe pose solution
mp_pose = medp.solutions.pose
mp_draw = medp.solutions.drawing_utils
pose = mp_pose.Pose()

def process_video(uploaded_file):
    # read the uploaded video file
    video_bytes = uploaded_file.read()

    # create a video stream object from the uploaded video file
    cap = cv2.VideoCapture(video_bytes)

    # create a video writer object for saving the processed video
    codec = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter('output.mp4', codec, cap.get(cv2.CAP_PROP_FPS),
                                   (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    # read each frame/image from capture object
    while True:
        ret, img = cap.read()
        # resize image/frame so we can accommodate it on our screen
        if img is None:
            break
        else:
            img = cv2.resize(img, (600, 400))

        # do Pose detection
        results = pose.process(img)
        # draw the detected pose on original video/ live stream
        mp_draw.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                               mp_draw.DrawingSpec((255, 0, 0), 2, 2),
                               mp_draw.DrawingSpec((255, 0, 255), 2, 2)
                               )

        # Extract and draw pose on plain white image
        h, w, c = img.shape  # get shape of original frame
        opImg = np.zeros([h, w, c])  # create blank image with original frame size
        opImg.fill(255)  # set white background. put 0 if you want to make it black

        # draw extracted pose on black white image
        mp_draw.draw_landmarks(opImg, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                               mp_draw.DrawingSpec((255, 0, 0), 2, 2),
                               mp_draw.DrawingSpec((255, 0, 255), 2, 2)
                               )

        # concatenate the original and result videos horizontally
        final_frame = np.concatenate((img, opImg), axis=1)

        # write the concatenated frame to the output video
        output_video.write(final_frame)

        # display the original and result videos on Streamlit
        st.image(final_frame, channels="BGR")

        # press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # release the capture object and video writer object
    cap.release()
    output_video.release()

# upload the video file from Streamlit
uploaded_file = st.sidebar.file_uploader("Upload your file here")

if uploaded_file:
    process_video(uploaded_file)
