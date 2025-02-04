import unittest
import requests

from unittest.mock import patch

from youtube_to_md import is_valid_youtube_url, is_accessible_youtube_url


class TestYouTubeFunctions(unittest.TestCase):

    def test_is_valid_youtube_url(self):
        # Valid YouTube URLs with different formats and query strings
        valid_urls = [
            "https://www.youtube.com/watch?v=DFYRQ_zQ-gk&feature=featured",
            "https://www.youtube.com/watch?v=DFYRQ_zQ-gk",
            "http://www.youtube.com/watch?v=DFYRQ_zQ-gk",
            "//www.youtube.com/watch?v=DFYRQ_zQ-gk",
            "www.youtube.com/watch?v=DFYRQ_zQ-gk",
            "https://youtube.com/watch?v=DFYRQ_zQ-gk",
            "http://youtube.com/watch?v=DFYRQ_zQ-gk",
            "//youtube.com/watch?v=DFYRQ_zQ-gk",
            "youtube.com/watch?v=DFYRQ_zQ-gk",
            "https://m.youtube.com/watch?v=DFYRQ_zQ-gk",
            "http://m.youtube.com/watch?v=DFYRQ_zQ-gk",
            "//m.youtube.com/watch?v=DFYRQ_zQ-gk",
            "m.youtube.com/watch?v=DFYRQ_zQ-gk",
            "https://www.youtube.com/v/DFYRQ_zQ-gk?fs=1&hl=en_US",
            "http://www.youtube.com/v/DFYRQ_zQ-gk?fs=1&hl=en_US",
            "//www.youtube.com/v/DFYRQ_zQ-gk?fs=1&hl=en_US",
            "www.youtube.com/v/DFYRQ_zQ-gk?fs=1&hl=en_US",
            "youtube.com/v/DFYRQ_zQ-gk?fs=1&hl=en_US",
            "https://www.youtube.com/embed/DFYRQ_zQ-gk?autoplay=1",
            "https://www.youtube.com/embed/DFYRQ_zQ-gk",
            "http://www.youtube.com/embed/DFYRQ_zQ-gk",
            "//www.youtube.com/embed/DFYRQ_zQ-gk",
            "www.youtube.com/embed/DFYRQ_zQ-gk",
            "https://youtube.com/embed/DFYRQ_zQ-gk",
            "http://youtube.com/embed/DFYRQ_zQ-gk",
            "//youtube.com/embed/DFYRQ_zQ-gk",
            "youtube.com/embed/DFYRQ_zQ-gk",
            "https://youtu.be/DFYRQ_zQ-gk?t=120",
            "https://youtu.be/DFYRQ_zQ-gk",
            "http://youtu.be/DFYRQ_zQ-gk",
            "//youtu.be/DFYRQ_zQ-gk",
            "youtu.be/DFYRQ_zQ-gk",
        ]

        for url in valid_urls:
            self.assertTrue(is_valid_youtube_url(url))

        # Invalid URLs
        invalid_urls = [
            "https://www.youtube.com/HamdiKickProduction?v=DFYRQ_zQ-gk",
            "https://www.example.com/watch?v=DFYRQ_zQ-gk",
            "https://youtu.be/",
            "https://youtube.com/",
        ]

        for url in invalid_urls:
            self.assertFalse(is_valid_youtube_url(url))

    def test_is_accessible_youtube_url(self):
        # Simulate a successful response
        # Mock requests.head to return status code 200
        with patch("requests.head") as mock_head:
            mock_head.return_value.status_code = 200
            self.assertTrue(is_accessible_youtube_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ"))

        # Simulate a response with a non-200 status code
        with patch("requests.head") as mock_head:
            mock_head.return_value.status_code = 404
            self.assertFalse(is_accessible_youtube_url("https://www.youtube.com/watch?v=nonexistent_video"))

        # Simulate a network exception
        with patch("requests.head", side_effect=requests.RequestException):
            self.assertFalse(is_accessible_youtube_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ"))


if  __name__ == "__main__":
    unittest.main()