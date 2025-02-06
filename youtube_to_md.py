import sys
import os
import argparse
import yt_dlp

# Constants
OUTPUT_FOLDER = "downloads"
AUDIO_FORMAT = "m4a"

def download_audio_from_youtube(url, output_folder=OUTPUT_FOLDER, audio_format=AUDIO_FORMAT, quiet=True):
    os.makedirs(output_folder, exist_ok=True)

    # Whisper handles M4A and OPUS efficiently without conversion
    # MP3 requires extra decoding, making it slightly less efficient for speech recognition.
    ydl_opts = {
        "format": f"bestaudio[ext={audio_format}]",
        "outtmpl": f"{output_folder}/%(title)s.%(ext)s",
        "noplaylist": True,
        "quiet": quiet,  # Suppress unnecessary output
        "no_warnings": True,  # Suppress warnings
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)  # Try to get video info without downloading
            if not info or ("entries" in info and not info["entries"]):
                print("Error: No valid video found at this URL.")
                return False

            ydl.download([url])
        return True
    except yt_dlp.utils.DownloadError as e:
        print(f"An error occurred during the audio download: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="YouTube video to Markdown article converter")
    parser.add_argument("url", type=str, help="The YouTube video URL to convert")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    success = download_audio_from_youtube(args.url, quiet=not args.verbose)
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()