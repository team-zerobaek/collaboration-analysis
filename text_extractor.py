import os
import subprocess
import sys
import time
from whisper import load_model
from pyannote.audio import Pipeline
from huggingface_hub import login

# Function to extract audio from video
def extract_audio(video_file, audio_file, start_time=None, end_time=None):
    command = ['ffmpeg', '-y', '-i', video_file, '-q:a', '0', '-map', 'a']

    if start_time and start_time != "None":
        command.extend(['-ss', start_time])
    if end_time and end_time != "None":
        command.extend(['-to', end_time])

    command.append(audio_file)

    subprocess.run(command, check=True)

# Function to transcribe audio using Whisper model
def transcribe_audio(audio_file):
    model = load_model("large")
    result = model.transcribe(audio_file)
    return result['text'], result['segments']

# Function to perform speaker diarization using pyannote.audio
def perform_speaker_diarization(audio_file, token, num_speakers=5):
    login(token=token)
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=token)
    diarization = pipeline(audio_file, min_speakers=num_speakers, max_speakers=num_speakers)

    # Get the speaker turns
    speaker_turns = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speaker_turns.append((turn.start, turn.end, speaker))

    return speaker_turns

# Main function
def main():
    start_time_global = time.time()

    video_file = 'records\\meetingrecord_2nd_teamzerobaek_proejct4.mp4'
    audio_file = 'audio.wav'

    start_time = None
    end_time = None
    num_speakers = 5

    if len(sys.argv) > 1:
        start_time = sys.argv[1]
    if len(sys.argv) > 2:
        end_time = sys.argv[2]
    if len(sys.argv) > 3:
        num_speakers = int(sys.argv[3])

    print(f"Set start time: {start_time}, end time: {end_time}, number of speakers: {num_speakers}")

    print("Starting audio extraction...")
    start_time_audio = time.time()
    extract_audio(video_file, audio_file, start_time, end_time)
    end_time_audio = time.time()
    print(f"Audio extraction completed in {end_time_audio - start_time_audio:.2f} seconds")

    print("Loading Whisper model...")
    start_time_transcription = time.time()
    transcription, segments = transcribe_audio(audio_file)
    end_time_transcription = time.time()
    print(f"Transcription completed in {end_time_transcription - start_time_transcription:.2f} seconds")

    print("Loading pyannote pipeline...")
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    start_time_diarization = time.time()
    diarization = perform_speaker_diarization(audio_file, hf_token, num_speakers)
    end_time_diarization = time.time()
    print(f"Speaker diarization completed in {end_time_diarization - start_time_diarization:.2f} seconds")

    print("Saving transcription with speaker labels...")
    with open('transcription.txt', 'w', encoding='utf-8') as f:
        for start, end, speaker in diarization:
            segment_text = " ".join([word['text'] for word in segments if start <= word['start'] < end])
            f.write(f"Speaker {speaker}: {segment_text}\n")
    print("Transcription saved to transcription.txt")

    end_time_global = time.time()
    print(f"Total time spent: {end_time_global - start_time_global:.2f} seconds")

if __name__ == "__main__":
    main()
