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

# Load variables from .env file
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Constants
OUTPUT_FOLDER = "downloads"
AUDIO_FORMAT_DOWNLOAD = "m4a"
AUDIO_FORMAT_CONVERTED = "mp3"
GROQ_MODEL_WHISPER = "whisper-large-v3"  # Whisper model on Groq
GROQ_MODEL = "llama3-8b-8192"  # LLaMA 3.3 model on Groq
LOG_FILE = "youtube_to_md.log"
MAX_TOKENS_PER_CHUNK = 4000
CHUNK_LENGTH_AUDIO_MS = 10 * 60 * 1000  # 10 minutes
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
    try:
        with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
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
    title = unicodedata.normalize("NFKC", title)
    title = re.sub(r"[<>:\"/\\|?*]", "", title)  # Windows restrictions
    title = re.sub(r"\s+", "_", title).strip("_")  # Replace spaces with underscores
    return f"{title[:max_length]}_{video_id}"


def download_audio(url, metadata, output_folder=OUTPUT_FOLDER, audio_format=AUDIO_FORMAT_DOWNLOAD):
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
        "extractor_retries": 5
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
                logging.error(f"Downloaded file not found before renaming: {downloaded_file}")
                return None  # Prevent renaming if the file is missing
            os.rename(downloaded_file, expected_file_name)
            return expected_file_name
    except yt_dlp.utils.DownloadError as e:
        logging.error(f"Download failed: {e}")
        return None
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return None


def convert_audio_to_mp3(audio_file):
    mp3_file = audio_file.rsplit(".", 1)[0] + f".{AUDIO_FORMAT_CONVERTED}"
    AudioSegment.from_file(audio_file).export(mp3_file, format=AUDIO_FORMAT_CONVERTED)
    os.remove(audio_file)
    return mp3_file


def split_audio(audio_file):
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
    client = groq.Client(api_key=GROQ_API_KEY)
    chunks = split_audio(audio_file)
    transcript = ""
    detected_languages = []

    for chunk in chunks:
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
    if not GROQ_API_KEY:
        logging.critical("Missing API Key!")
        return None

    client = groq.Client(api_key=GROQ_API_KEY)
    transcript_chunks = [transcript[i: i + MAX_TOKENS_PER_CHUNK] for i in
                         range(0, len(transcript), MAX_TOKENS_PER_CHUNK)]
    formatted_chunks = []

    # os.makedirs("formatted_segments", exist_ok=True)  # Ensure folder exists

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

        # with open(f"formatted_segments/formatted_{index}.md", "w", encoding="utf-8") as fmt_file:
        #     fmt_file.write(formatted_text)

        formatted_chunks.append(formatted_text)

    return "\n\n".join(formatted_chunks)


def save_transcript_to_markdown(formatted_text, metadata, url, output_folder="articles"):
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
    parser = argparse.ArgumentParser(description="YouTube video to Markdown converter")
    parser.add_argument("url", type=str, help="YouTube video URL")
    args = parser.parse_args()

    metadata = get_video_metadata(args.url)
    if not metadata:
        sys.exit(1)

    audio_file = download_audio(args.url, metadata)
    if not audio_file:
        sys.exit(1)

    mp3_file = convert_audio_to_mp3(audio_file)
    transcript, _ = transcribe_audio(mp3_file)
    formatted_text = format_transcript_with_llama(transcript)
    save_transcript_to_markdown(formatted_text, metadata, args.url)


if __name__ == "__main__":
    main()
