import argparse
import re
import requests
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
    Checks if a given YouTube URL is accessible by sending an HTTP request.

    The function sends a HEAD request to the provided URL to determine if it is reachable.
    A valid and accessible YouTube URL should return an HTTP status code of 200.

    Returns:
        bool: True if the URL is accessible (returns HTTP 200), False otherwise.

    Examples:
        >>> is_accessible_youtube_url("https://www.youtube.com/watch?v=DFYRQ_zQ-gk")
        True
        >>> is_accessible_youtube_url("https://www.youtube.com/watch?v=invalid_video_id")
        False
        >>> is_accessible_youtube_url("https://www.example.com/video")
        False
    """

    try:
        response = requests.get(url, allow_redirects=True, timeout=5)
        if response.status_code == 200 and "ytInitialPlayerResponse" in response.text:
            return True
        return False
    except (requests.RequestException, requests.Timeout):
        return False

def validate_youtube_url(url):
    """
    Validates whether a given URL is both a correctly formatted YouTube link and accessible.

    The function first checks if the URL matches a valid YouTube video format.
    Then, it sends a request to verify whether the video is reachable.

    Returns:
        bool: True if the URL is a valid YouTube video link and accessible, False otherwise.

    Examples:
        >>> validate_youtube_url("https://www.youtube.com/watch?v=DFYRQ_zQ-gk")
        True
        >>> validate_youtube_url("https://www.youtube.com/watch?v=invalid_video_id")
        False
        >>> validate_youtube_url("https://www.example.com/video")
        False
    """
    if is_valid_youtube_url(url) and is_accessible_youtube_url(url):
        return True
    return False

def download_audio_from_youtube(url, output_folder="downloads", audio_format="mp3"):
    """
    This function downloads the audio from a YouTube video given its URL.
    """
    if not validate_youtube_url(url):
        print("Invalid YouTube URL or URL is not accessible.")
        return False

    os.makedirs(output_folder, exist_ok=True)

    ydl_opts = {
        "format": "bestaudio/best",
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
    parser.add_argument("output", type=str, default="downloads", help="The output folder for the downloaded audio")

    args = parser.parse_args()

    success = download_audio_from_youtube(args.url, args.output)

    if success:
        print("Download successful!")
    else:
        print("Download failed!")

if __name__ == "__main__":
    main()