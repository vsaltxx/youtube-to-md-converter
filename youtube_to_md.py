import sys
import os
import re
import argparse
import yt_dlp
import whisper
import groq
import logging

from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Constants
OUTPUT_FOLDER = "downloads"
AUDIO_FORMAT = "m4a"
GROQ_MODEL = "llama3-8b-8192"    # LLaMA 3.3 model on Groq
LOG_FILE = "youtube_to_md.log"

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Load Whisper model once
WHISPER_MODEL = whisper.load_model("base")

def get_video_metadata(url):
    logging.info("Fetching video metadata...")
    ydl_opts = {"quiet": True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        return {
            "title": info.get("title", "Unknown Title"),
            "duration": info.get("duration", 0),
            "uploader": info.get("uploader", "Unknown"),
        }

def safe_filename(title, video_id):
    """Removes illegal characters from filenames and appends video ID."""
    sanitized_title = re.sub(r'[<>:"/\\|?*]', "", title)
    return f"{sanitized_title}_{video_id}"

def download_audio_from_youtube(url, output_folder=OUTPUT_FOLDER, audio_format=AUDIO_FORMAT, quiet=True):
    logging.info(f"Downloading audio from {url}...")
    os.makedirs(output_folder, exist_ok=True)

    # Whisper handles M4A and OPUS efficiently without conversion
    # MP3 requires extra decoding, making it slightly less efficient for speech recognition.
    ydl_opts = {
        "format": f"bestaudio[ext={audio_format}]",
        "outtmpl": f"{output_folder}/%(title)s.%(ext)s",
        "noplaylist": True,
        "quiet": quiet,
        "no_warnings": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            if not info or ("entries" in info and not info["entries"]):
                logging.warning("No valid video found at this URL.")
                return False

            # Get actual downloaded file path
            downloaded_file = ydl.prepare_filename(info).replace(".webm", f".{audio_format}")
            logging.info(f"Download successful: {downloaded_file}")
            return downloaded_file
    except yt_dlp.utils.DownloadError as e:
        logging.error(f"Download error: {e}")
        return False

def transcribe_audio(audio_file):
    """Transcribes an audio file using OpenAI Whisper."""
    logging.info(f"Transcribing audio: {audio_file}...")
    try:
        result = WHISPER_MODEL.transcribe(audio_file, fp16=False)
        return result["text"] if "text" in result else None
    except Exception as e:
        logging.error(f"Transcription error: {e}")
        return None

def process_text_with_llama(transcript):
    """Uses LLaMA 3.3 via Groq API to process and format the transcript."""

    if not GROQ_API_KEY:
        logging.critical("GROQ_API_KEY is missing! Please set it in your .env file.")
        sys.exit(1)

    client = groq.Client(api_key=GROQ_API_KEY)

    prompt = f"""
    Format this transcript into a well-structured Markdown article.
    - Add headings, bullet points, and summaries
    - Keep the text readable and structured

    Transcript:
    {transcript}

    Formatted Markdown:
    """
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except groq.RateLimitError as e:
        logging.error(f"Rate limit error: {e}")
        return None
    except groq.AuthenticationError as e:
        logging.error(f"Authentication error: {e}")
        return None
    except groq.APIError as e:
        logging.error(f"API error: {e}")
        return None
    except Exception as e:
        logging.error(f"Error calling Groq API: {e}")
        return None

def format_transcript_to_markdown(transcript, metadata, url, output_folder="articles"):
    """Formats the transcript into a Markdown article and saves it to a file."""
    os.makedirs(output_folder, exist_ok=True)

    video_id = url.split("v=")[-1].split("&")[0]            # Extract Video ID
    formatted_text = process_text_with_llama(transcript)    # Process transcript using LLaMA 3.3

    if formatted_text is None:
        logging.critical("Error: Failed to format transcript using LLaMA 3.3.")
        sys.exit(1)

    md_content = f"# {metadata['title']}\n\n"
    md_content += f"**Uploader:** {metadata['uploader']}\n"
    md_content += f"**Duration:** {metadata['duration']} seconds\n"
    md_content += f"**Original Video:** [{metadata['title']}]({url})\n\n"
    md_content += "---\n\n"
    md_content += formatted_text

    output_file = os.path.join(output_folder, f"{safe_filename(metadata['title'], video_id)}.md")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(md_content)

    logging.info(f"Markdown file saved: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="YouTube video to Markdown article converter")
    parser.add_argument("url", type=str, help="The YouTube video URL to convert")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    audio_file = download_audio_from_youtube(args.url, quiet=not args.verbose)
    if not audio_file:
        sys.exit(1)

    transcript = transcribe_audio(audio_file)
    if not transcript:
        logging.critical("Transcription failed.")
        sys.exit(1)

    metadata = get_video_metadata(args.url)
    format_transcript_to_markdown(transcript, metadata, args.url)

if __name__ == "__main__":
    main()
