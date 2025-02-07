import sys
import os
import re
import argparse
import yt_dlp
import whisper

# Constants
OUTPUT_FOLDER = "downloads"
AUDIO_FORMAT = "m4a"

# Load Whisper model once
WHISPER_MODEL = whisper.load_model("base")

def get_video_metadata(url):
    ydl_opts = {"quiet": True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        return {
            "title": info.get("title", "Unknown Title"),
            "duration": info.get("duration", 0),
            "uploader": info.get("uploader", "Unknown"),
        }

def safe_filename(title):
    """Removes illegal characters from filenames."""
    return re.sub(r'[<>:"/\\|?*]', "", title)

def download_audio_from_youtube(url, output_folder=OUTPUT_FOLDER, audio_format=AUDIO_FORMAT, quiet=True):
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
                print("Error: No valid video found at this URL.")
                return False

            # Get actual downloaded file path
            downloaded_file = ydl.prepare_filename(info).replace(".webm", f".{audio_format}")
            return downloaded_file
    except yt_dlp.utils.DownloadError as e:
        print(f"An error occurred during the audio download: {e}")
        return False

def transcribe_audio(audio_file):
    """Transcribes an audio file using OpenAI Whisper."""
    print("Transcribing audio...")
    try:
        result = WHISPER_MODEL.transcribe(audio_file, fp16=False)
        return result["text"] if "text" in result else None
    except Exception as e:
        print(f"Error during transcription: {e}")
        return None

def format_transcript_to_markdown(transcript, metadata, url, output_folder="articles"):
    os.makedirs(output_folder, exist_ok=True)

    md_content = f"# {metadata['title']}\n\n"
    md_content += f"**Uploader:** {metadata['uploader']}\n"
    md_content += f"**Duration:** {metadata['duration']} seconds\n"
    md_content += f"**Original Video:** [{metadata['title']}]({url})\n\n"
    md_content += "---\n\n"

    for paragraph in transcript.split("\n"):
        if paragraph.strip():
            md_content += f"{paragraph.strip()}\n\n"

    output_file = os.path.join(output_folder, f"{safe_filename(metadata['title'])}.md")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"Markdown file saved: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="YouTube video to Markdown article converter")
    parser.add_argument("url", type=str, help="The YouTube video URL to convert")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    audio_file = download_audio_from_youtube(args.url, quiet=not args.verbose)
    if not audio_file:
        sys.exit(1)

    print(f"Downloaded audio to: {audio_file}")

    transcript = transcribe_audio(audio_file)
    if not transcript:
        print("Transcription failed.")
        sys.exit(1)

    metadata = get_video_metadata(args.url)
    format_transcript_to_markdown(transcript, metadata, args.url)

if __name__ == "__main__":
    main()
