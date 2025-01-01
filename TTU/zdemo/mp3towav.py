import os
from pydub import AudioSegment

# Define the directory containing the audio files
directory = "/sailhome/duyy/data/AudioLLM/TTU/zdemo/"

# Loop through all files in the directory
for filename in os.listdir(directory):
    if filename.endswith(".mp3"):
        # Define the full path to the .mp3 file
        mp3_path = os.path.join(directory, filename)
        
        # Define the path for the converted .wav file
        wav_filename = filename.replace(".mp3", ".wav")
        wav_path = os.path.join(directory, wav_filename)
        
        # Load the .mp3 file and convert it to .wav
        audio = AudioSegment.from_mp3(mp3_path)
        audio.export(wav_path, format="wav")
        
        print(f"Converted {filename} to {wav_filename}")

print("All files have been converted.")
