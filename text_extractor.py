import os
import subprocess
import sys
import time
from whisper import load_model
from pyannote.audio import Pipeline, Model
from huggingface_hub import login

# Function to extract audio from video
def extract_audio(video_file, audio_file, start_time=None, end_time=None):
    command = ['ffmpeg', '-y', '-i', video_file, '-q:a', '0', '-map', 'a']
    if start_time and end_time:
        command.extend(['-ss', start_time, '-to', end_time])
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
    model = Model.from_pretrained("pyannote/segmentation", use_auth_token=token)

    # Voice Activity Detection
    from pyannote.audio.pipelines import VoiceActivityDetection
    vad_pipeline = VoiceActivityDetection(segmentation=model)
    VAD_HYPER_PARAMETERS = {
      "onset": 0.684, "offset": 0.577,
      "min_duration_on": 0.181,
      "min_duration_off": 0.037
    }
    vad_pipeline.instantiate(VAD_HYPER_PARAMETERS)
    vad = vad_pipeline(audio_file)

    # Overlapped Speech Detection
    from pyannote.audio.pipelines import OverlappedSpeechDetection
    osd_pipeline = OverlappedSpeechDetection(segmentation=model)
    OSD_HYPER_PARAMETERS = {
      "onset": 0.448, "offset": 0.362,
      "min_duration_on": 0.116,
      "min_duration_off": 0.187
    }
    osd_pipeline.instantiate(OSD_HYPER_PARAMETERS)
    osd = osd_pipeline(audio_file)

    # Speaker Diarization
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=token)
    diarization = pipeline(audio_file, min_speakers=num_speakers, max_speakers=num_speakers)

    # Get the speaker turns
    speaker_turns = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speaker_turns.append((turn.start, turn.end, speaker))

    return speaker_turns

# Main function
def main():
    video_file = 'records/meetingrecord_1st_teamzerobaek_proejct4.mp4'
    audio_file = 'audio.wav'

    start_time = None
    end_time = None
    num_speakers = 5

    if len(sys.argv) > 1 and sys.argv[1] != 'None':
        start_time = sys.argv[1]
    if len(sys.argv) > 2 and sys.argv[2] != 'None':
        end_time = sys.argv[2]
    if len(sys.argv) > 3:
        num_speakers = int(sys.argv[3])

    start_time_total = time.time()
    print(f"Starting audio extraction from {start_time} to {end_time}...")
    start_time_extract = time.time()
    extract_audio(video_file, audio_file, start_time, end_time)
    end_time_extract = time.time()
    print(f"Audio extraction completed in {end_time_extract - start_time_extract:.2f} seconds")

    print("Loading Whisper model...")
    start_time_transcribe = time.time()
    transcription, segments = transcribe_audio(audio_file)
    end_time_transcribe = time.time()
    print(f"Transcription completed in {end_time_transcribe - start_time_transcribe:.2f} seconds")

    print("Loading pyannote pipeline...")
    start_time_diarize = time.time()
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    diarization = perform_speaker_diarization(audio_file, hf_token, num_speakers)
    end_time_diarize = time.time()
    print(f"Speaker diarization completed in {end_time_diarize - start_time_diarize:.2f} seconds")

    print("Saving transcription with speaker labels...")
    start_time_save = time.time()
    with open('transcription.txt', 'w', encoding='utf-8') as f:
        for start, end, speaker in diarization:
            segment_text = " ".join([word['text'] for word in segments if start <= word['start'] < end])
            f.write(f"Speaker {speaker}: {segment_text}\n")
    end_time_save = time.time()
    print(f"Transcription saved to transcription.txt in {end_time_save - start_time_save:.2f} seconds")

    end_time_total = time.time()
    print(f"Total time spent: {end_time_total - start_time_total:.2f} seconds")

if __name__ == "__main__":
    main()
