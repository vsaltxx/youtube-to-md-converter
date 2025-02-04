import re

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

