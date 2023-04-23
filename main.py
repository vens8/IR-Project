import streamlit as st
import requests
import time
import moviepy.editor as mp
from io import StringIO
from PIL import Image
import os
import nltk
import matplotlib.pyplot as plt

nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('words')


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
			with open(f'assets/transcripts/{_id}.txt', 'w') as f:
				f.write(polling_response.json()['text'])
			print(f'Transcript saved to assets/transcripts/{_id}.txt')

			# Perform Named Entity Recognition (NER)
			text = polling_response.json()['text']
			print('text', text)
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

			# Display the chart on Streamlit
			st.pyplot(fig)
			break

		elif polling_response.json()['status'] == 'error':
			raise Exception("Transcription failed. Make sure a valid API key has been used.")
		else:
			print("Transcription queued or processing ...")
			# time.sleep(5)
	return


# @st.cache_data
# def change_text(data):
#     data=data.lower()
#     data=nltk.word_tokenize(data)
#     english_stopwords = stopwords.words()
#     tokens_wo_stopwords = [t for t in data if t not in english_stopwords]
#     data=tokens_wo_stopwords
#     lst=[]
#     for i in data:
#         if(i.isalpha()):
#             lst.append(i)   
#     data= lst
#     return " ".join(data)    

# @st.cache_data
# def analyse_text():
#     file_name=""
#     data=""
#     for file in os.listdir(os.getcwd()):
#         if file.endswith(".txt"):
#             file_name=file
#             # file_name=file_name+".txt"
#             # print(file_name)
#             with open(file_name, 'r') as file:
#                 data = file.read() 

# final_data=change_text(data)
# word_cloud = WordCloud(collocations = False, background_color = 'white').generate(final_data)
# # Display the generated Word Cloud
# plt.imshow(word_cloud, interpolation='bilinear')
# plt.axis("off")
# plt.show()

@st.cache_data
def print_plots():
	dependency_plot = Image.open('dependency_plot.jpg')
	image_waveform = Image.open("image_waveform.png")
	wordcloud = Image.open("wordcloud.png")
	st.image(dependency_plot, caption='Dependency Plot')
	st.image(image_waveform, caption="Image Waveform")
	st.image(wordcloud, caption="WordCloud")

	# analyse_text()
	# data=data.lower()
	# data=nltk.word_tokenize(data)
	# english_stopwords = stopwords.words()
	# tokens_wo_stopwords = [t for t in data if t not in english_stopwords]
	# data=tokens_wo_stopwords
	# lst=[]
	# for i in data:
	#     if(i.isalpha()):
	#         lst.append(i)   
	# data= lst
	# return " ".join(data) 


@st.cache_data
def analyse_audio():
	audio_file = open('converted.wav', 'rb')  # enter the filename with filepath
	audio_bytes = audio_file.read()  # reading the file
	st.audio(audio_bytes, format='audio/ogg')  # displaying the audio


st.set_page_config(page_title="Emoture - An Emotion Recognition App")
st.title("Emoture - An Emotion Recognition App")
st.write("---")
st.sidebar.title("Navigation")

uploaded_file = st.sidebar.file_uploader("Upload your file here")

if uploaded_file:
	process_video(uploaded_file)

options = st.sidebar.radio("Page To Navigate", options=["Home", "Plots", "Audio"])

if options == "Home":
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
if options == "Plots":
	if (uploaded_file is None):
		st.header("Not uploaded")
	else:
		st.header("Uploaded")
		print_plots()

elif options == "Audio":
	if (uploaded_file is None):
		st.header("Not uploaded")
	else:
		st.header("Uploaded")
	analyse_audio()