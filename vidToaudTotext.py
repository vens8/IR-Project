import requests
import time
import moviepy.editor as mp


clip = mp.VideoFileClip(r"video.mp4") 
clip.audio.write_audiofile(r"converted.wav")

# Endpoints for uploading and transcribing audio
UPLOAD_ENDPOINT = "https://api.assemblyai.com/v2/upload"
TRANSCRIPTION_ENDPOINT = "https://api.assemblyai.com/v2/transcript"

# Variables for requests
api_key = "dc737c08289441578a2f68b04d404f74"
headers = {"authorization": api_key, "content-type": "application/json"}

# Generator to read an audio file
def read_file(filename):
   with open(filename, 'rb') as _file:
       while True:
           data = _file.read(5242880)
           if not data:
               break
           yield data

# Upload the audio file to AssemblyAI 
upload_response = requests.post(UPLOAD_ENDPOINT, headers=headers, data=read_file('converted.wav'))
# data = read_file('converted.wav')
audio_url = upload_response.json()["upload_url"]

# Request a transcription from AssemblyAI
transcript_request = {'audio_url': audio_url}
transcript_response = requests.post(TRANSCRIPTION_ENDPOINT, json=transcript_request, headers=headers)
_id = transcript_response.json()["id"]

# Wait for transcription to complete, and then save the resulting transcript
while True:
    # Get updated transcription info
    polling_response = requests.get(TRANSCRIPTION_ENDPOINT + "/" + _id, headers=headers)

    # If the transcription is complete, save it to a .txt file
    if polling_response.json()['status'] == 'completed':
        with open(f'{_id}.txt', 'w') as f:
            f.write(polling_response.json()['text'])
        print('Transcript saved to', _id, '.txt')
        break
    # If the transcription has failed, raise an Exception
    elif polling_response.json()['status'] == 'error':
        raise Exception("Transcription failed. Make sure a valid API key has been used.")
    # Otherwise, print that the transcription is in progress
    else:
        print("Transcription queued or processing ...")
        time.sleep(5)
