import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# Read API key from environment variable
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Transcribing function
def transcribe_audio(file_path, page_number):
    with open(file_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="whisper-1", 
            file=audio_file
        )

    return {
        "page_number": page_number,
        "content": transcription.text
    }

# Main
audio_files = ["mp3/audio1.mp3"]  # List of audio files
json_data = []

for i, file_path in enumerate(audio_files, start=1):
    json_data.append(transcribe_audio(file_path, page_number=i))

# Save to JSON file
with open("json/transcriptions.json", "w", encoding="utf-8") as json_file:
    json.dump(json_data, json_file, ensure_ascii=False, indent=4)

print("Transcriptions saved to transcriptions.json")
