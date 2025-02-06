import unittest
from unittest.mock import patch, MagicMock
from youtube_to_md import download_audio_from_youtube

class TestYouTubeFunctions(unittest.TestCase):

    @patch("yt_dlp.YoutubeDL")  # Mock yt_dlp to prevent real downloads
    def test_download_audio_valid_url(self, mock_ytdlp):
        """Test downloading audio from a valid YouTube URL."""
        mock_instance = MagicMock()
        mock_ytdlp.return_value.__enter__.return_value = mock_instance
        mock_instance.extract_info.return_value = {"title": "Sample Video"}  # Simulate video info
        mock_instance.download.return_value = None  # Simulate successful download

        valid_url = "https://www.youtube.com/watch?v=DFYRQ_zQ-gk"
        result = download_audio_from_youtube(valid_url)

        self.assertTrue(result)  # The function should return True

    @patch("yt_dlp.YoutubeDL")
    def test_download_audio_invalid_url(self, mock_ytdlp):
        """Test downloading audio from an invalid YouTube URL."""
        mock_instance = MagicMock()
        mock_ytdlp.return_value.__enter__.return_value = mock_instance
        mock_instance.extract_info.return_value = None  # Simulate no valid video
        mock_instance.download.side_effect = Exception("DownloadError")  # Simulate failure

        invalid_url = "https://youtube.com/"
        result = download_audio_from_youtube(invalid_url)

        self.assertFalse(result)  # The function should return False

if __name__ == "__main__":
    unittest.main()
