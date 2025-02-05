import argparse
import re
import os

import yt_dlp


def is_valid_youtube_url(url):
    """
    Checks if a given URL matches one of the supported YouTube formats.

    Supported formats:
        - Standard YouTube watch links (e.g., https://www.youtube.com/watch?v=VIDEO_ID)
        - Shortened links (e.g., https://youtu.be/VIDEO_ID)
        - Embedded links (e.g., https://www.youtube.com/embed/VIDEO_ID)
        - Mobile links (e.g., https://m.youtube.com/watch?v=VIDEO_ID)
        - Links with query parameters (e.g., ?t=120 or &feature=featured)

    Returns:
        bool: True if the URL matches one of the supported formats, False otherwise.

    Examples:
        >>> is_valid_youtube_url("https://www.youtube.com/watch?v=abc123")
        True
        >>> is_valid_youtube_url("https://example.com/video")
        False
    """

    youtube_url_pattern = (
         r'^((?:https?:)?\/\/)?((?:www|m)\.)?((?:youtube\.com\/(?:watch\?v=|embed\/|v\/)|youtu\.be\/))([\w\-]+)(\S+)?$'
    )
    return bool(re.match(youtube_url_pattern, url))

def is_accessible_youtube_url(url):
    """
    Checks if the given YouTube URL is accessible using yt-dlp.

    Returns:
        bool: True if the video exists, False otherwise.
    """
    ydl_opts = {
        "quiet": True,          # Suppress unnecessary output
        "no_warnings": True,    # Suppress warnings
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False) # Extract video information without downloading
            return True if info else False
    except yt_dlp.utils.DownloadError:
        return False # Video is not accessible
    except Exception as e:
        print(f"An error occurred while checking the video accessibility: {e}")
        return False # Any other error

def download_audio_from_youtube(url, output_folder="downloads", audio_format="m4a"):
    os.makedirs(output_folder, exist_ok=True)
    # Whisper handles M4A and OPUS efficiently without conversion
    # MP3 requires extra decoding, making it slightly less efficient for speech recognition.
    ydl_opts = {
        "format": "bestaudio[ext=m4a]",
        "extractaudio": True,
        "audioformat": audio_format,
        "outtmpl": f"{output_folder}/%(title)s.%(ext)s",
        "quiet": False,
        "noplaylist": True
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
            print("Audio download completed successfully!")
        return True
    except Exception as e:
        print(f"An error occurred during the audio download: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="YouTube URL Validator")
    parser.add_argument("url", type=str, help="The YouTube video URL to convert")

    args = parser.parse_args()

    if not is_valid_youtube_url(args.url):
        print("Invalid YouTube URL. Please provide a valid YouTube video URL.")
        return

    if not is_accessible_youtube_url(args.url):
        return

    success = download_audio_from_youtube(args.url)
    if success:
        print("Download successful!")
    else:
        print("Download failed!")

if __name__ == "__main__":
    main()