import shutil
import sys
import os
import re
import time
import argparse
import unicodedata
import yt_dlp
import groq
import logging
import httpx
from pydub import AudioSegment
from dotenv import load_dotenv
from langdetect import detect
from collections import Counter
from tqdm import tqdm

# Load variables from .env file
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Constants
OUTPUT_FOLDER = "downloads"
AUDIO_FORMAT_DOWNLOAD = "m4a"
AUDIO_FORMAT_CONVERTED = "mp3"
GROQ_MODEL_WHISPER = "whisper-large-v3"     # Whisper model on Groq
GROQ_MODEL = "llama3-8b-8192"               # LLaMA 3.3 model on Groq
LOG_FILE = "youtube_to_md.log"
MAX_TOKENS_PER_CHUNK = 4000
CHUNK_LENGTH_AUDIO_MS = 10 * 60 * 1000      # 10 minutes
MAX_RETRIES = 3

TOO_MANY_REQUESTS_ERROR = 429
RATE_LIMIT_WAIT = 10  # seconds

# Logging Setup
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def setup_output_folder(folder):
    os.makedirs(folder, exist_ok=True)


def get_video_metadata(url):
    """Fetch metadata for the YouTube video."""
    try:
        with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True}) as ydl:
            info = ydl.extract_info(url, download=False)
            return {
                "title": info.get("title", "Unknown Title"),
                "duration": info.get("duration", 0),
                "uploader": info.get("uploader", "Unknown"),
                "video_id": info.get("id", "unknown")
            }
    except yt_dlp.utils.DownloadError as e:
        logging.error(f"Failed to get metadata: {e}")
        return None
    except Exception as e:
        logging.error(f"Unknown error: {e}")
        return None


def safe_filename(title, video_id, max_length=100):
    """Generate a safe filename for the Markdown file."""
    title = unicodedata.normalize("NFKC", title)
    title = re.sub(r"[<>:\"/\\|?*]", "", title)  # Windows restrictions
    title = re.sub(r"\s+", "_", title).strip("_")  # Replace spaces with underscores
    return f"{title[:max_length]}_{video_id}"


def download_audio(url, metadata, output_folder=OUTPUT_FOLDER, audio_format=AUDIO_FORMAT_DOWNLOAD):
    """Download the audio from the YouTube video."""
    setup_output_folder(output_folder)
    safe_name = safe_filename(metadata["title"], metadata["video_id"])
    expected_file_name = os.path.join(output_folder, f"{safe_name}.{audio_format}")

    ydl_opts = {
        "format": f"bestaudio[ext={audio_format}]",
        "outtmpl": f"{output_folder}/%(title)s.%(ext)s",
        "noplaylist": True,
        "retries": 10,
        "nocheckcertificate": True,
        "force_generic_extractor": True,
        "extractor_retries": 5,
        "quiet": True,          # Hide logs
        "no_warnings": True     # Suppress warnings
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            downloaded_file = ydl.prepare_filename(info).replace(".webm", f".{audio_format}")

            # If expected file already exists, remove it to prevent rename conflicts
            if os.path.exists(expected_file_name):
                logging.warning(f"File {expected_file_name} already exists. Removing to avoid conflicts.")
                os.remove(expected_file_name)

            # Ensure the downloaded file exists before renaming
            if not os.path.exists(downloaded_file):
                print("Error: Download failed. Possible causes:")
                print("- The video may be restricted.")
                print("- You may need a VPN for this region.")
                print("- Your internet connection might be unstable.")
                logging.error(f"Downloaded file not found before renaming: {downloaded_file}")
                return None  # Prevent renaming if the file is missing
            os.rename(downloaded_file, expected_file_name)
            print(f"Download complete: {expected_file_name}")
            return expected_file_name
    except yt_dlp.utils.DownloadError as e:
        logging.error(f"Download failed: {e}")
        print(f"Download Error: {e}")
        print("Try using a VPN or checking if the video is accessible.")
        return None
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        print(f"File Error: {e}")
        print("Ensure there is enough disk space available.")
        return None
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        print(f"Unexpected Error: {e}")
        print("Try running the script again or updating yt-dlp.")
        return None


def convert_audio_to_mp3(audio_file):
    """Converts an audio file to MP3 format and removes the original file to save space."""
    mp3_file = audio_file.rsplit(".", 1)[0] + f".{AUDIO_FORMAT_CONVERTED}"
    AudioSegment.from_file(audio_file).export(mp3_file, format=AUDIO_FORMAT_CONVERTED)
    os.remove(audio_file) # Remove the original audio file
    return mp3_file


def split_audio(audio_file):
    """Splits the audio file into smaller chunks for transcription."""
    if not os.path.exists(audio_file):
        logging.error(f"File not found: {audio_file}")
        return []

    audio = AudioSegment.from_file(audio_file)
    chunks = []

    for i in range(0, len(audio), CHUNK_LENGTH_AUDIO_MS):
        chunk_path = f"{audio_file[:-4]}_part_{i // CHUNK_LENGTH_AUDIO_MS}.{AUDIO_FORMAT_CONVERTED}"
        audio[i: i + CHUNK_LENGTH_AUDIO_MS].export(chunk_path, format=AUDIO_FORMAT_CONVERTED)
        chunks.append(chunk_path)
    logging.info(f"Split into {len(chunks)} segments")
    return chunks


def detect_dominant_language(text):
    """Detects the dominant language in the text and filters out non-dominant languages."""
    sentences = text.split("\n")
    lang_counts = Counter()
    detected_sentences = []

    for sentence in sentences:
        if sentence.strip():
            try:
                lang = detect(sentence)
                lang_counts[lang] += 1
                detected_sentences.append((sentence, lang))
            except:
                detected_sentences.append((sentence, "unknown"))

    dominant_language = lang_counts.most_common(1)[0][0] if lang_counts else "unknown"
    filtered_sentences = [s for s, lang in detected_sentences if lang == dominant_language]
    return "\n".join(filtered_sentences), dominant_language


def transcribe_audio(audio_file):
    """Transcribes the audio file using the Whisper model on Groq."""
    client = groq.Client(api_key=GROQ_API_KEY)
    chunks = split_audio(audio_file)
    transcript = ""
    detected_languages = []

    for chunk in tqdm(chunks, desc="Transcribing"):
        for attempt in range(MAX_RETRIES):
            try:
                with open(chunk, "rb") as f:
                    response = client.audio.transcriptions.create(
                        model=GROQ_MODEL_WHISPER,
                        file=f,
                        response_format="json"
                    )
                # Extract transcription and language safely
                segment_transcript = getattr(response, "text", "")
                segment_language = getattr(response, "language", "unknown")
                transcript += segment_transcript + "\n"
                detected_languages.append(segment_language)
                os.remove(chunk)
                break # Exit loop if successful
            except FileNotFoundError as e:
                logging.error(f"File not found: {e}")
                return None, None
            except httpx.HTTPStatusError as e:
                if e.response.status_code == TOO_MANY_REQUESTS_ERROR:  # Rate limit exceeded
                    wait_time = (2 ** attempt) * RATE_LIMIT_WAIT  # Exponential backoff
                    logging.warning(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logging.error(f"Error transcribing {chunk}: {e}")
                    break

    refined_transcript, dominant_language = detect_dominant_language(transcript)
    return refined_transcript.strip(), dominant_language


def format_transcript_with_llama(transcript):
    """Formats the transcript into Markdown using the LLaMA model."""
    client = groq.Client(api_key=GROQ_API_KEY)
    transcript_chunks = [transcript[i: i + MAX_TOKENS_PER_CHUNK] for i in
                         range(0, len(transcript), MAX_TOKENS_PER_CHUNK)]
    formatted_chunks = []

    for index, chunk in enumerate(transcript_chunks):
        prompt = f"""
                You are an expert Markdown formatter. Format the following text properly:
                {chunk}
                ## Output (Only return the formatted text, do NOT add any intro messages):
                """

        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
        )

        formatted_text = response.choices[0].message.content.strip()
        formatted_text = re.sub(
            r"^\s*(Here is the formatted text:|Transcript:|Formatted text:|Here is your formatted Markdown:)", "",
            formatted_text, flags=re.IGNORECASE).strip()

        formatted_chunks.append(formatted_text)

    return "\n\n".join(formatted_chunks)


def save_transcript_to_markdown(formatted_text, metadata, url, output_folder="articles"):
    """Saves the transcript in Markdown format to a file."""
    setup_output_folder(output_folder)
    clean_title = safe_filename(metadata["title"], metadata["video_id"])
    output_file = os.path.join(output_folder, f"{clean_title}.md")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(
            f"# {metadata['title']}\n\n"
            f"**Uploader:** {metadata['uploader']}\n\n"
            f"**Duration:** {metadata['duration']} seconds\n\n"
            f"**Original Video:** [{metadata['title']}]({url})\n\n"
            f"---\n\n{formatted_text}"
        )
    logging.info(f"Markdown file saved: {output_file}")


def main():
    """
    Main function to convert YouTube video to Markdown.
    Usage: python youtube_to_md.py <YouTube URL>
    """
    parser = argparse.ArgumentParser(description="YouTube video to Markdown converter")
    parser.add_argument("url", type=str, help="YouTube video URL")
    args = parser.parse_args()

    # Check API key
    if not GROQ_API_KEY:
        print("Error: Missing GROQ API key. Set it in your .env file.")
        sys.exit(1)

    start_time = time.time()
    print("\nStarting YouTube to Markdown conversion...")

    t0 = time.time()
    print("Fetching video metadata...")
    metadata = get_video_metadata(args.url)
    if not metadata:
        print("Error: Failed to retrieve video metadata. Exiting.")
        sys.exit(1)
    print(f"Video Found: {metadata['title']} by {metadata['uploader']} ({time.time() - t0:.2f}s)")

    print("Downloading audio...")
    audio_file = download_audio(args.url, metadata)
    if not audio_file:
        print("Error: Audio download failed. Exiting.")
        sys.exit(1)
    print(f"Audio downloaded: {audio_file} ({time.time() - t0:.2f}s)")

    mp3_file = convert_audio_to_mp3(audio_file)

    t0 = time.time()
    print("Transcribing audio... This may take a few minutes.")
    transcript, _ = transcribe_audio(mp3_file)
    print(f"Transcription complete ({time.time() - t0:.2f}s)")

    t0 = time.time()
    print("Formatting transcript into Markdown...")
    formatted_text = format_transcript_with_llama(transcript)
    print(f"Formatting complete ({time.time() - t0:.2f}s)")

    print("Saving Markdown file...")
    save_transcript_to_markdown(formatted_text, metadata, args.url)

    # Final Cleanup: Remove downloads folder
    if os.path.exists("downloads"):
        shutil.rmtree("downloads")  # Delete the entire folder

    total_time = time.time() - start_time
    print(f"\nConversion complete! Your article is ready in 'articles' folder. (Total Time: {total_time:.2f}s)")

if __name__ == "__main__":
    main()
