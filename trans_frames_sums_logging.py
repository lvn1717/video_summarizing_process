import os
import numpy as np
import pandas as pd
import subprocess
import whisper
import glob
import base64
from datetime import datetime
import time
import logging


from openai import OpenAI

# Set your API key (alternatively, you can set the OPENAI_API_KEY environment variable)
MY_OPENAI_API_KEY =   # <-- Replace with your API key
client = OpenAI(api_key=MY_OPENAI_API_KEY)

# =============================================================================
# Logger Setup
# =============================================================================
logger = logging.getLogger('video_processing')
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')


file_handler = logging.FileHandler('video_processing.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


script_start_time = datetime.now()
print(f"Script started at: {script_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
logger.info(f"Script started at: {script_start_time.strftime('%Y-%m-%d %H:%M:%S')}")

#  Define All Directories
BASE_DIR = r"E:\house5mp4\HOUSE5_batch_23_mp4"
METADATA_FNAME = os.path.join(BASE_DIR, "METADATA.csv")
VIDEO_DIR = r"E:\house5mp4\HOUSE5_batch_23_mp4"
TRANSCRIPT_JSON = os.path.join(BASE_DIR, "pres_ad_whisptranscripts_json")
TRANSCRIPT_TSV = os.path.join(BASE_DIR, "pres_ad_whisptranscripts_tsv")
TRANSCRIPT_TXT = os.path.join(BASE_DIR, "pres_ad_whisptranscripts_txt")
FRAME_SPEECH = os.path.join(BASE_DIR, "keyframes_speechcentered")
FRAME_INTERVAL = os.path.join(BASE_DIR, "keyframes_regintervals")
FRAME_DESCRIPTIONS_SPEECH = os.path.join(BASE_DIR, "GPT_frame_descriptions_speechcentered")
FRAME_DESCRIPTIONS_INTERVAL = os.path.join(BASE_DIR, "GPT_frame_descriptions_regintervals")
OUTPUT_SUMMARY = os.path.join(BASE_DIR, "GPT_video_summaries")

#  Ensure Directories Exist
for directory in [TRANSCRIPT_JSON, TRANSCRIPT_TSV, TRANSCRIPT_TXT, FRAME_SPEECH, FRAME_INTERVAL,
                  FRAME_DESCRIPTIONS_SPEECH, FRAME_DESCRIPTIONS_INTERVAL, OUTPUT_SUMMARY]:
    os.makedirs(directory, exist_ok=True)
logger.info("Ensured all required directories exist.")


logger.info("Loading Whisper model...")
whisper_model = whisper.load_model("large-v3")
logger.info("Whisper model loaded.")


def log(msg):
    print(f"{msg}")
    logger.info(msg)


def transcribe_videos():
    log("Starting transcription process...")

    video_files = glob.glob(os.path.join(VIDEO_DIR, "*.mp4"))
    existing_transcripts = set(os.path.splitext(os.path.basename(f))[0] for f in glob.glob(os.path.join(TRANSCRIPT_JSON, "*.json")))

    for video in video_files:
        video_id = os.path.splitext(os.path.basename(video))[0]

        if video_id in existing_transcripts:
            log(f"Skipping {video_id} (Already transcribed)")
            continue

        try:
            log(f"Transcribing {video_id}...")
            transcription = whisper_model.transcribe(video, fp16=False)

            with open(os.path.join(TRANSCRIPT_TXT, f"{video_id}.txt"), "w", encoding="utf-8", errors="replace") as text_file:
                text_file.write(transcription['text'])
           
            
            from whisper.utils import get_writer
            json_writer = get_writer("json", TRANSCRIPT_JSON)
            json_writer(transcription, f"{video_id}.json")

            tsv_writer = get_writer("tsv", TRANSCRIPT_TSV)
            tsv_writer(transcription, f"{video_id}.tsv")

            log(f" Transcribed: {video_id}")

        except Exception as e:
            log(f" Error transcribing {video_id}: {e}")
            logger.error(f"Error transcribing {video_id}: {e}")

transcribe_videos()


def extract_keyframes():
    log(" Extracting keyframes...")

    try:
        metadata_df = pd.read_csv(METADATA_FNAME)
    except Exception as e:
        log(f" Error reading metadata file {METADATA_FNAME}: {e}")
        logger.error(f"Error reading metadata file {METADATA_FNAME}: {e}")
        return

    for _, row in metadata_df.iterrows():
        video_filename = row['FILENAME']
        video_path = os.path.join(VIDEO_DIR, video_filename)
        video_id = os.path.splitext(video_filename)[0]

        if not os.path.exists(video_path):
            log(f"video file not found: {video_path}")
            logger.error(f"Video file not found: {video_path}")
            continue

        #  Get Video Duration
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', video_path],
            capture_output=True, text=True
        )
        duration_str = result.stdout.strip()

        if not duration_str:
            # If no duration is returned, log the ffprobe error message and skip this file.
            log(f" ffprobe error for {video_path}: {result.stderr.strip()}")
            logger.error(f"ffprobe error for {video_path}: {result.stderr.strip()}")
            continue

        try:
            video_duration = float(duration_str) * 1000  # Convert to milliseconds
        except ValueError as e:
            log(f" Could not convert duration '{duration_str}' to float for {video_path}: {e}")
            logger.error(f"Could not convert duration '{duration_str}' to float for {video_path}: {e}")
            continue

        log(f"Extracting frames for {video_id} (Duration: {video_duration / 1000:.2f} seconds)")

        #Regular interval keyframes (every 3 seconds)
        for timestamp in range(3000, 180000, 3000):
            if timestamp >= video_duration:
                break
            frame_path = os.path.join(FRAME_INTERVAL, f"{video_id}_{timestamp}.jpg")
            if not os.path.exists(frame_path):
                subprocess.run([
                    'ffmpeg', '-y', '-hide_banner', '-loglevel', 'warning',
                    '-ss', str(timestamp / 1000.), '-i', video_path,
                    '-frames:v', '1', frame_path
                ])
                log(f"Extracted frame {frame_path}")
                logger.info(f"Extracted frame {frame_path}")

extract_keyframes()


def describe_frames():
    print("\n Generating frame descriptions...")
    logger.info("Starting frame descriptions generation...")

    try:
        metadata_df = pd.read_csv(METADATA_FNAME)
    except Exception as e:
        print(f" Error reading metadata file {METADATA_FNAME}: {e}")
        logger.error(f"Error reading metadata file {METADATA_FNAME}: {e}")
        return

    for _, row in metadata_df.iterrows():
        video_id = os.path.splitext(row['FILENAME'])[0]

        #Read transcript with encoding fallback
        transcript_path = os.path.join(TRANSCRIPT_TXT, f"{video_id}.txt")
        transcript = "No transcript available."
        if os.path.exists(transcript_path):
            try:
                with open(transcript_path, "r", encoding="utf-8", errors="replace") as text_file:
                    transcript = text_file.read()
            except Exception as e:
                print(f" Error reading transcript for {video_id}: {e}")
                logger.error(f"Error reading transcript for {video_id}: {e}")

        #Process frames for speech & interval
        for frame_folder, output_folder in [(FRAME_SPEECH, FRAME_DESCRIPTIONS_SPEECH),
                                            (FRAME_INTERVAL, FRAME_DESCRIPTIONS_INTERVAL)]:
            for frame_path in glob.glob(os.path.join(frame_folder, f"{video_id}_*.jpg")):
                output_path = os.path.join(output_folder, f"{os.path.basename(frame_path)}.txt")

                if os.path.exists(output_path):
                    continue

                try:
                    with open(frame_path, "rb") as img_file:
                        encoded_frame = base64.b64encode(img_file.read()).decode("utf-8")

                    prompt = f"Describe this video frame in no more than 15 words. Context: {transcript}"

                    
                    response = client.chat.completions.create(
                        model="gpt-4-turbo",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=100
                    )
                    

                    output_text = response.choices[0].message.content
                    with open(output_path, "w", encoding="utf-8") as outfile:
                        outfile.write(output_text)

                    print(f"Described frame {frame_path} → {output_path}")
                    logger.info(f"Described frame {frame_path} saved to {output_path}")

                except Exception as e:
                    print(f" OpenAI API Error: {e}")
                    logger.error(f"OpenAI API Error while describing frame {frame_path}: {e}")

describe_frames()


def summarize_videos():
    print("\n Generating video summaries...")
    logger.info("Starting video summarization process...")

    try:
        metadata_df = pd.read_csv(METADATA_FNAME)
    except Exception as e:
        print(f" Error reading metadata file {METADATA_FNAME}: {e}")
        logger.error(f"Error reading metadata file {METADATA_FNAME}: {e}")
        return

    if metadata_df.empty:
        print("METADATA.csv is empty no videos to process.")
        logger.warning("METADATA.csv is empty. No videos to process.")
        return

    for _, row in metadata_df.iterrows():
        video_filename = row['FILENAME']
        party = row.get('PARTY', 'Unknown Party')
        election_year = str(row.get('ELECTION', 'Unknown Year'))
        candidate = f"{row.get('FIRST_NAME', '')} {row.get('LAST_NAME', '')}".strip() or "Unknown Candidate"

        #Check if summary exists
        output_path = os.path.join(OUTPUT_SUMMARY, f"{video_filename}.txt")
        if os.path.exists(output_path):
            print(f"Skipping {video_filename} (Already summarized)")
            logger.info(f"Skipping {video_filename} as it is already summarized.")
            continue

        #Load transcript
        transcript_path = os.path.join(TRANSCRIPT_TXT, f"{video_filename}.txt")
        transcript = "No transcript available."
        if os.path.exists(transcript_path):
            try:
                with open(transcript_path, "r", encoding="utf-8") as file:
                    transcript = file.read().strip() or "Transcript exists but is empty."
            except Exception as e:
                print(f"Error reading transcript for {video_filename}: {e}")
                logger.error(f"Error reading transcript for {video_filename}: {e}")

        #Collect frame descriptions
        frametimes, framedescriptions = [], []

        # Speech-centered descriptions
        for frame_path in glob.glob(os.path.join(FRAME_DESCRIPTIONS_SPEECH, f"{os.path.splitext(video_filename)[0]}_*.txt")):
            try:
                frametime = float(frame_path.split("_")[-1].split(".")[0])  
                with open(frame_path, "r", encoding="utf-8") as file:
                    framedescription = file.read().strip()
                frametimes.append(frametime)
                framedescriptions.append(framedescription)
            except ValueError:
                print(f"Skipping invalid frame file: {frame_path}")
                logger.warning(f"Skipping invalid frame file: {frame_path}")

        # Interval descriptions
        for frame_path in glob.glob(os.path.join(FRAME_DESCRIPTIONS_INTERVAL, f"{os.path.splitext(video_filename)[0]}_*.txt")):
            try:
                frametime = float(frame_path.split("_")[-1].split(".")[0])
                with open(frame_path, "r", encoding="utf-8") as file:
                    framedescription = file.read().strip()
                frametimes.append(frametime)
                framedescriptions.append(framedescription)
            except ValueError:
                print(f"Skipping invalid frame file: {frame_path}")
                logger.warning(f"Skipping invalid frame file: {frame_path}")

        print(f"Processing video: {video_filename}")
        logger.info(f"Processing video: {video_filename}")
        print(f"Transcript length: {len(transcript)} characters")
        logger.info(f"Transcript length for {video_filename}: {len(transcript)} characters")
        print(f"Number of frames found: {len(framedescriptions)}")
        logger.info(f"Number of frames found for {video_filename}: {len(framedescriptions)}")
        print("Speech-centered frames found:", glob.glob(os.path.join(FRAME_DESCRIPTIONS_SPEECH, f"{video_filename}_*.txt")))
        print("Interval frames found:", glob.glob(os.path.join(FRAME_DESCRIPTIONS_INTERVAL, f"{video_filename}_*.txt")))
        logger.info(f"Speech-centered frames: {glob.glob(os.path.join(FRAME_DESCRIPTIONS_SPEECH, f'{os.path.splitext(video_filename)[0]}_*.txt'))}")
        logger.info(f"Interval frames: {glob.glob(os.path.join(FRAME_DESCRIPTIONS_INTERVAL, f'{os.path.splitext(video_filename)[0]}_*.txt'))}")

        #Skip if no frames found
        if not frametimes or not framedescriptions:
            print(f"No valid frames found for {video_filename}. Skipping...")
            logger.error(f"No valid frames found for {video_filename}. Skipping...")
            continue

        #Sort frame descriptions by time
        sorted_indices = np.argsort(frametimes)
        framedescriptions_sorted = np.asarray(framedescriptions)[sorted_indices]

        #Construct GPT Prompt
        prompt = (
            f"Provide a 50-word summary of a political TV ad. "
            f"This ad was for the {election_year} presidential campaign of {party} candidate {candidate}. "
            f"The transcript is:\n{transcript}\n\n"
            f"The ad video contains these scenes:\n"
            + "\n".join([f"{idx + 1}: {desc}" for idx, desc in enumerate(framedescriptions_sorted)])
        )

        print("\n\nGPT PROMPT:\n", prompt, "\n")
        logger.info(f"GPT Prompt for {video_filename}: {prompt}")

        try:
            for _ in range(3):  # Retry up to 3 times
                try:
                    
                    response = client.chat.completions.create(
                        model="gpt-4-turbo",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=100
                    )
          
                    result = response.choices[0].message.content.strip()
                    break  # Exit retry loop if successful
                except Exception as e:
                    print(f" OpenAI API error: {e}. Retrying in 5 seconds...")
                    logger.error(f"OpenAI API error for {video_filename}: {e}. Retrying in 5 seconds...")
                    time.sleep(5)  # Wait and retry
            else:
                print(f" Final OpenAI API failure for {video_filename}. Skipping...")
                logger.error(f"Final OpenAI API failure for {video_filename}. Skipping...")
                continue  # Skip if all retries fail

            #Save summary
            with open(output_path, "w", encoding="utf-8") as outfile:
                outfile.write(result)

            print(f"Summarized {video_filename} → {output_path}")
            logger.info(f"Summarized {video_filename} saved to {output_path}")

        except Exception as e:
            print(f"Unexpected error generating summary for {video_filename}: {e}")
            logger.error(f"Unexpected error generating summary for {video_filename}: {e}")

summarize_videos()

print("\n**ALL DONE!**")
logger.info("All processing steps completed successfully.")

# =============================================================================
# Script End
# =============================================================================
script_end_time = datetime.now()
total_duration = script_end_time - script_start_time

print(f"\nScript completed at: {script_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Total execution time: {total_duration}")
logger.info(f"Script completed at: {script_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
logger.info(f"Total execution time: {total_duration}")
