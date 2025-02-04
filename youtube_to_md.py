import re
import requests

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
        response = requests.head(url, allow_redirects=True, timeout=5)
        return response.status_code == 200
    except (requests.RequestException, requests.Timeout):
        return False

