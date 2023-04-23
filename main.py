import streamlit as st
import requests
import time
import moviepy.editor as mp
from sklearn.feature_extraction.text import TfidfVectorizer
from io import StringIO
from PIL import Image
import os 
import cv2
import cv2
import mediapipe as medp
# initialize mediapipe pose solution
mp_pose = medp.solutions.pose
mp_draw = medp.solutions.drawing_utils
pose = mp_pose.Pose()
import numpy as np
import pandas as pd
import nltk
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from wordcloud import WordCloud
import librosa.display
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('words')
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer

uploaded_file=None


@st.cache_data
def process_video(uploaded_file):
    clip = mp.VideoFileClip(uploaded_file.name) 
    clip.audio.write_audiofile(r"converted.wav")
    UPLOAD_ENDPOINT = "https://api.assemblyai.com/v2/upload"
    TRANSCRIPTION_ENDPOINT = "https://api.assemblyai.com/v2/transcript"
    api_key = "dc737c08289441578a2f68b04d404f74"
    headers = {"authorization": api_key, "content-type": "application/json"}

    def read_file(filename):
       with open(filename, 'rb') as _file:
           while True:
               data = _file.read(5242880)
               if not data:
                   break
               yield data

    upload_response = requests.post(UPLOAD_ENDPOINT, headers=headers, data=read_file('converted.wav'))
    audio_url = upload_response.json()["upload_url"]
    transcript_request = {'audio_url': audio_url}
    transcript_response = requests.post(TRANSCRIPTION_ENDPOINT, json=transcript_request, headers=headers)
    _id = transcript_response.json()["id"]

    while True:
        polling_response = requests.get(TRANSCRIPTION_ENDPOINT + "/" + _id, headers=headers)
        if polling_response.json()['status'] == 'completed':
            transcript_file = 'transcript.txt'
            if os.path.exists(transcript_file):
                os.remove(transcript_file)
            with open(transcript_file, 'w') as f:
                f.write(polling_response.json()['text'])
            print('Transcript saved to', transcript_file)
            
            # Perform Named Entity Recognition (NER)
            text = polling_response.json()['text']
            break
            
        elif polling_response.json()['status'] == 'error':
            raise Exception("Transcription failed. Make sure a valid API key has been used.")
        else:
            print("Transcription queued or processing ...")
            # time.sleep(5)    
    return


@st.cache_data
def change_text(data):
    data=data.lower()
    data=nltk.word_tokenize(data)
    english_stopwords = stopwords.words('english')
    tokens_wo_stopwords = [t for t in data if t not in english_stopwords]
    data=tokens_wo_stopwords
    lst=[]
    for i in data:
        if(i.isalpha()):
            lst.append(i)   
    data= lst
    return " ".join(data)    

@st.cache_data
def analyse_text():
    file_name=""
    data=""
    for file in os.listdir(os.getcwd()):
        if file.endswith(".txt"):
            file_name=file
            with open(file_name, 'r') as file:
                data = file.read() 
    
    final_data=change_text(data)
    return final_data
from PIL import Image

@st.cache_data
def NER(text):
    entities = nltk.chunk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(text)))
    entity_count = nltk.FreqDist([ent.label() for ent in entities if hasattr(ent, 'label')])
    # Create a bar chart to display the NER results
    fig, ax = plt.subplots()
    print(entity_count.keys())
    print(entity_count.values())
    ax.bar(entity_count.keys(), entity_count.values())
    ax.set_xlabel('Entity Types')
    ax.set_ylabel('Count')
    ax.set_title('Named Entity Recognition Results')
    
    image = Image.open('C:/Users/Sumit Soni/Desktop/IR/IR-Project/NER_Image.jpeg')

    st.image(image, caption='Naned Entity Recognition Plot', use_column_width=True)

@st.cache_data
def word_cloud(final_data):
    word_cloud = WordCloud(collocations = False, background_color = 'white').generate(final_data)
    fig, ax = plt.subplots()
    ax.imshow(word_cloud, interpolation='bilinear')
    ax.axis("off")
    # Use st.pyplot to display the figure in Streamlit app
    st.pyplot(fig)

@st.cache_data
def audio_waveform(audio_file):

    signal, sr = librosa.load(audio_file, sr=None)
    fig, ax = plt.subplots()
    librosa.display.waveplot(signal, sr=sr, ax=ax)
    st.pyplot(fig)


@st.cache_data
def TF_IDF(text):
    # Tokenize the text
    tokens = nltk.word_tokenize(text)
    
    # Remove stopwords and punctuations
    tokens = [token.lower() for token in tokens if token.lower() not in stopwords.words('english') and token.isalnum()]
    
    # Calculate TF-IDF scores
    tfidf_vectorizer = TfidfVectorizer(tokenizer=lambda doc: doc, lowercase=False)
    tfidf_matrix = tfidf_vectorizer.fit_transform([tokens])
    feature_names = tfidf_vectorizer.get_feature_names()
    tfidf_scores = pd.DataFrame(tfidf_matrix.T.todense(), index=feature_names, columns=["tfidf"])
    tfidf_scores = tfidf_scores.sort_values(by=["tfidf"], ascending=False).reset_index()
    
    # Select top 10 words based on TF-IDF scores
    top_words = tfidf_scores.head(10)["index"].tolist()
    
    # Generate co-occurrence matrix for top words
    vectorizer = CountVectorizer(vocabulary=top_words)
    doc_matrix = vectorizer.fit_transform([text])
    vocab = vectorizer.get_feature_names()
    co_occurrence_matrix = doc_matrix.T.dot(doc_matrix).toarray()

    # Plot the co-occurrence matrix
    fig, ax = plt.subplots()
    im = ax.imshow(co_occurrence_matrix, cmap='coolwarm')
    ax.set_xticks(np.arange(len(vocab)))
    ax.set_yticks(np.arange(len(vocab)))
    ax.set_xticklabels(vocab)
    
    # Display the letters of the words in labels along y axis vertically
    yticklabels = ax.set_yticklabels(vocab)
    for label in yticklabels:
        label.set_rotation(0)
        label.set_horizontalalignment('right')
        
    ax.set_xlabel('Words')
    ax.set_ylabel('Words')
    ax.set_title('Co-occurrence matrix for top words')
    cbar = ax.figure.colorbar(im, ax=ax)

    # Display the plots in Streamlit
    st.write("Top 10 words based on TF-IDF scores:")
    st.write(top_words)
    st.pyplot(fig)
 
@st.cache_data
def print_plots():
    text=analyse_text()
    
    st.title('Word Cloud')
    word_cloud(text)
    
    st.title("Audio Waveform")
    audio_waveform('converted.wav')
    
    st.title("NER")
    NER(text)

    # st.title("Co-Ocurrence Matrix")
    # Co_Occurence(text)
    
    st.title("TF IDF")
    TF_IDF(text)

@st.cache_data
def Final_Analysis():
    
    text=analyse_text()
    blob = TextBlob(text)
    # polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    st.header("Sentiment Subjectivity")
    st.write("Sentimental subjectivity refers to the emotional response that a person has towards a particular subject or object. It involves the expression of personal feelings, attitudes, and beliefs that can be influenced by past experiences, cultural background, and individual differences. Sentiment analysis is a natural language processing technique that involves the use of computational methods to identify, extract, and quantify the sentiment expressed in a piece of text. The score generated from sentiment analysis can be used to determine the overall emotional tone of a text, which can be positive, negative, or neutral. Sentiment analysis scores can be used for various applications, such as social media monitoring, customer feedback analysis, and market research. For example, companies can use sentiment analysis to analyze customer reviews and feedback to determine the overall sentiment towards their products or services. Additionally, sentiment analysis can be used by political campaigns to monitor the sentiment of voters towards a particular candidate or issue.")
    st.write(subjectivity)

    st.header("Sentiment Analysis (From Audio)")
    st.write("man_angry")

@st.cache_data
def analyse_audio():
    audio_file = open('converted.wav','rb') #enter the filename with filepath
    audio_bytes = audio_file.read() #reading the file
    st.audio(audio_bytes, format='audio/ogg') #displaying the audio
    
st.set_page_config(page_title="Emoture - An Emotion Recognition App")
st.title("Emoture - An Emotion Recognition App")
st.write("---")
st.sidebar.title("Navigation")

@st.cache_data
def process_posture():
    global uploaded_file
    # create a temporary file to store the uploaded video
    temp_file = open(uploaded_file.name, 'wb')
    temp_file.write(uploaded_file.getbuffer())

    # take video input for pose detection
    cap = cv2.VideoCapture(temp_file.name)

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

        # normalize the pixel values of the final frame
        final_frame = final_frame / 255.0

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

@st.cache_data
def make_skelaton():
    st.title("Pose Estimation Demo")
    st.write("This demo uses MediaPipe to estimate human poses in a video or live stream.")
    process_posture()
  
  
  
uploaded_file=st.sidebar.file_uploader("Upload your file here")

if uploaded_file:
    process_video(uploaded_file)
    
    
options=st.sidebar.radio("Page To Navigate", options=["Home", "Plots", "Audio", "Final Analysis"])      

if options=="Home":
    st.header("Home")
    st.image("image_1.jpg", use_column_width=True)
    st.markdown("<h2 id='about'>About</h2>", unsafe_allow_html=True)
    st.write("""
        This app is a simple user interface for our IR Project - An Integrated Approach to Context-aware Emotion Recognition through Posture and Speech Modulation Analysis. 
        Through our ML models, one's emotional state can be inferred by analysing their posture and speech modulation, while also being contextually aware.
    """)
    st.markdown("## What is this app?")
    st.write("""
        This app is a simple user interface for our IR Project - An Integrated Approach to Context-aware Emotion Recognition through Posture and Speech Modulation Analysis. 
        Through our ML models, one's emotional state can be inferred by analysing their posture and speech modulation, while also being contextually aware.
    """)
    st.markdown("## Why use this app?")
    st.write("""
        There are several benefits to using this app to play your videos:
        - No need to download any software
        - Works on any device with a web browser
        - Plug and play, easy and quick to use.
    """)
    st.markdown("## How to use this app?")
    st.write("""
        1. Select 'Upload Video' from the menu.
        2. Choose an MP4 video file to upload.
        3. Wait for the video to upload and process (this may take a while for large files).
        4. Wait for the processing and view your results!
        """)
    st.markdown("<h2 id='contact'>Contact</h2>", unsafe_allow_html=True)
    st.write("""
        If you have any questions or feedback about the app, please send an email to sumit20136@iiitd.ac.in.
    """)
if options=="Plots":
    if(uploaded_file is None):
        st.header("Not uploaded")
    else:
        # st.header("Uploaded")
        print_plots()
    
elif options=="Audio":
    if(uploaded_file is None):
        st.header("Not uploaded")
    else:
        # st.header("Uploaded")
        analyse_audio()
        
elif options=="Skelaton Frame":
    if(uploaded_file is None):
        st.header("Not uploaded")
    else:
        # st.header("Uploaded")
        make_skelaton()
        
        
elif options=="Final Analysis":
    if(uploaded_file is None):
        st.header("Not uploaded")
    else:
        # st.header("Uploaded")
        Final_Analysis()
        
    