import unittest
from unittest.mock import patch, MagicMock, mock_open

from youtube_to_md import (
    download_audio_from_youtube,
    transcribe_audio,
    process_text_with_llama,
)

class TestYouTubeToMarkdown(unittest.TestCase):


    @patch("yt_dlp.YoutubeDL")
    def test_download_audio_invalid_url(self, mock_ytdlp):
        """Test downloading audio from an invalid YouTube URL."""
        mock_instance = MagicMock()
        mock_ytdlp.return_value.__enter__.return_value = mock_instance
        mock_instance.extract_info.return_value = None  # No valid video
        mock_instance.download.side_effect = Exception("DownloadError")  # Simulate failure

        invalid_url = "https://youtube.com/"
        result = download_audio_from_youtube(invalid_url)

        self.assertFalse(result)  # Function should return False


    @patch("youtube_to_md.groq.Client")
    @patch("os.path.isfile")
    def test_transcribe_audio_file_not_found(self, mock_isfile, mock_groq_client):
        """Test transcribe_audio when the file does not exist"""

        # Mock file existence to return False
        mock_isfile.return_value = False

        # Call function
        transcript, detected_language = transcribe_audio("non_existent_file.m4a")

        # Assertions
        self.assertIsNone(transcript)
        self.assertIsNone(detected_language)

    @patch("groq.Client")
    def test_process_text_with_llama(self, mock_groq_client):
        """Test processing text with LLaMA (mock API call)."""
        mock_client_instance = MagicMock()
        mock_groq_client.return_value = mock_client_instance
        mock_client_instance.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="## Formatted Markdown"))]
        )

        result = process_text_with_llama("Sample transcript", "en")
        self.assertTrue(result.startswith("## Formatted Markdown"))  # Ensure it's formatted


if __name__ == "__main__":
    unittest.main()
