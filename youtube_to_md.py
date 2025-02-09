import sys
import os
import re
import argparse
import yt_dlp
import groq
import logging
import tiktoken
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Constants
OUTPUT_FOLDER = "downloads"
AUDIO_FORMAT = "m4a"
GROQ_MODEL_WHISPER = "whisper-large-v3"  # Whisper model on Groq
GROQ_MODEL = "llama3-8b-8192"    # LLaMA 3.3 model on Groq
LOG_FILE = "youtube_to_md.log"
MAX_TOKENS_PER_CHUNK = 4000  #Groq's API token limit is 6,000 tokens per request

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

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
    """Transcribes an audio file using Groq's OpenAI Whisper API with language auto-detection."""
    logging.info(f"Transcribing audio via Groq API: {audio_file}...")

    # Check if audio file exists
    if not os.path.isfile(audio_file):
        logging.error(f"Audio file not found: {audio_file}")
        return None, None # Return None for both transcript and detected language

    if not GROQ_API_KEY:
        logging.critical("GROQ_API_KEY is missing! Please set it in your .env file.")
        sys.exit(1)

    client = groq.Client(api_key=GROQ_API_KEY)

    # Send audio file to Groq's Whisper API
    try:
        with open(audio_file, "rb") as f:
            response = client.audio.transcriptions.create(
                model=GROQ_MODEL_WHISPER,
                file=f,
                response_format="json",
            )

        transcript = response.text if hasattr(response, "text") else None
        detected_language = response.language if hasattr(response, "language") else "unknown"

        logging.info(f"Detected language: {detected_language}")
        return transcript, detected_language
    except groq.RateLimitError as e:
        logging.error(f"Rate limit error: {e}")
        return None, None
    except groq.AuthenticationError as e:
        logging.error(f"Authentication error: {e}")
        return None, None
    except groq.APIError as e:
        logging.error(f"API error: {e}")
        return None, None
    except Exception as e:
        logging.error(f"Error calling Groq API: {e}")
        return None, None

def split_text_into_chunks(text):
    """Splits text into smaller chunks based on token limits."""
    encoder = tiktoken.get_encoding("cl100k_base")
    tokens = encoder.encode(text)

    chunks = []
    for i in range(0, len(tokens), MAX_TOKENS_PER_CHUNK):
        chunk_tokens = tokens[i:i + MAX_TOKENS_PER_CHUNK]
        chunk_text = encoder.decode(chunk_tokens)
        chunks.append(chunk_text)

    return chunks

def process_text_with_llama(transcript, detected_language):
    """Uses LLaMA 3.3 via Groq API to process and format the transcript."""

    if not GROQ_API_KEY:
        logging.critical("GROQ_API_KEY is missing! Please set it in your .env file.")
        sys.exit(1)

    client = groq.Client(api_key=GROQ_API_KEY)

    # Split transcript into chunks
    transcript_chunks = split_text_into_chunks(transcript)

    formatted_chunks = []
    for index, chunk in enumerate(transcript_chunks):
        logging.info(f"Processing chunk {index + 1}/{len(transcript_chunks)}...")

        prompt = f"""
        Format the following transcript into well-structured Markdown in its original language ({detected_language}).
        - **Keep all text segmentation and structure.**
        - **Use proper headings, bullet points, and speaker labels.**
        - **Preserve timestamps (if available) and important context.**
        - **Do NOT summarize or add conclusions.**
        
        Transcript:
        {chunk}
    
        Formatted Markdown:
        """

        try:
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
            formatted_chunks.append(response.choices[0].message.content.strip())

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

    # Combine all formatted chunks into one document
    return "\n\n".join(formatted_chunks)

def format_transcript_to_markdown(formatted_text, metadata, url, output_folder="articles"):
    """Formats the transcript into a Markdown file while keeping the original structure."""
    os.makedirs(output_folder, exist_ok=True)

    video_id = url.split("v=")[-1].split("&")[0]            # Extract Video ID

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

    transcript, detected_language = transcribe_audio(audio_file)
    if not transcript:
        logging.critical("Transcription failed.")
        sys.exit(1)

    metadata = get_video_metadata(args.url)
    formatted_transcript = process_text_with_llama(transcript, detected_language)
    if not formatted_transcript:
        logging.critical("Formatting failed.")
        sys.exit(1)

    format_transcript_to_markdown(formatted_transcript, metadata, args.url)

if __name__ == "__main__":
    main()
