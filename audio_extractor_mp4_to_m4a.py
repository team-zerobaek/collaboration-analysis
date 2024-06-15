import subprocess
import os

# Function to convert an MP4 video to an M4A audio file
def convert_mp4_to_m4a(video_file, output_audio_file, start_time=None, end_time=None):
    command = ['ffmpeg', '-y', '-i', video_file, '-vn', '-c:a', 'aac', '-b:a', '256k', '-map', 'a']

    # Add start and end times if provided
    if start_time:
        command.extend(['-ss', start_time])
    if end_time:
        command.extend(['-to', end_time])

    command.append(output_audio_file)

    # Run the ffmpeg command
    subprocess.run(command, check=True)
    print(f"Audio converted and saved as {output_audio_file}")

# Function to retrieve and display encoding type and sample rate
def get_audio_info(audio_file):
    command = ['ffmpeg', '-i', audio_file]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        output = result.stderr  # ffmpeg typically sends info to stderr

        # Extract encoding type and sample rate
        encoding_type = None
        sample_rate = None
        for line in output.split('\n'):
            if 'Audio:' in line:
                parts = line.split(',')
                encoding_type = parts[0].split('Audio: ')[1].strip()
                sample_rate = parts[1].strip()
                break

        print(f"Encoding type: {encoding_type}")
        print(f"Sample rate: {sample_rate}")
    except subprocess.CalledProcessError as e:
        print(f"Error retrieving audio info: {e.stderr}")

# Ensure paths are correctly formatted for Windows
input_video_file = os.path.normpath("records\\meetingrecord_5th_teamzerobaek_proejct4.mp4")
output_audio_file = os.path.normpath("audio\\meetingrecord_5th_teamzerobaek_proejct4.m4a")

# Convert entire video to M4A
convert_mp4_to_m4a(input_video_file, output_audio_file)

# Get audio info
get_audio_info(output_audio_file)

# Example with a specific range
# start_time = "00:01:00"  # Start time (HH:MM:SS)
# end_time = "00:02:00"  # End time (HH:MM:SS)
# convert_mp4_to_m4a(input_video_file, os.path.normpath("audio\\meetingrecord_8th_teamzerobaek_proejct4_range.m4a"), start_time, end_time)
# get_audio_info(os.path.normpath("audio\\meetingrecord_8th_teamzerobaek_proejct4_range.m4a"))
