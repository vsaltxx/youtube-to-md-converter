import unittest
import tiktoken
from unittest.mock import patch, MagicMock, mock_open

from youtube_to_md import (
    download_audio_from_youtube,
    transcribe_audio,
    process_text_with_llama,
    split_text_into_chunks,
)

from youtube_to_md import MAX_TOKENS_PER_CHUNK


class TestYouTubeToMarkdown(unittest.TestCase):

    @patch("yt_dlp.YoutubeDL")  # Mock yt_dlp to prevent real downloads
    def test_download_audio_valid_url(self, mock_ytdlp):
        """Test downloading audio from a valid YouTube URL."""
        mock_instance = MagicMock()
        mock_ytdlp.return_value.__enter__.return_value = mock_instance
        mock_instance.extract_info.return_value = {"title": "Sample Video"}
        mock_instance.download.return_value = None

        valid_url = "https://www.youtube.com/watch?v=DFYRQ_zQ-gk"
        result = download_audio_from_youtube(valid_url)

        self.assertTrue(result)  # Function should return a valid filename

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

    @patch("youtube_to_md.groq.Client")  # Mock the Groq API client
    @patch("builtins.open", new_callable=mock_open, read_data=b"dummy data")  # Mock file opening
    @patch("os.path.isfile")  # Mock file existence check
    def test_transcribe_audio_success(self, mock_isfile, mock_file, mock_groq_client):
        """Test transcribe_audio function with a valid response"""

        # Mock file existence
        mock_isfile.return_value = True

        # Mock API response
        mock_response = MagicMock()
        mock_response.text = "Mocked transcription result"
        mock_response.language = "en"

        # Mock API client
        mock_client_instance = mock_groq_client.return_value
        mock_client_instance.audio.transcriptions.create.return_value = mock_response

        # Call function
        transcript, detected_language = transcribe_audio("dummy_audio_file.m4a")

        # Assertions
        self.assertEqual(transcript, "Mocked transcription result")
        self.assertEqual(detected_language, "en")
        mock_file.assert_called_with("dummy_audio_file.m4a", "rb")  # Ensure the file is opened

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

    @patch("youtube_to_md.groq.Client")
    @patch("builtins.open", new_callable=mock_open, read_data=b"dummy data")
    @patch("os.path.isfile")
    def test_transcribe_audio_api_error(self, mock_isfile, mock_file, mock_groq_client):
        """Test transcribe_audio when Groq API raises an error"""

        # Mock file existence
        mock_isfile.return_value = True

        # Mock API client raising an API error
        mock_client_instance = mock_groq_client.return_value
        mock_client_instance.audio.transcriptions.create.side_effect = Exception("Mock API failure")

        # Call function
        transcript, detected_language = transcribe_audio("dummy_audio_file.m4a")

        # Assertions
        self.assertIsNone(transcript)
        self.assertIsNone(detected_language)
        mock_file.assert_called_with("dummy_audio_file.m4a", "rb")  # Ensure file is opened

    def test_split_text_into_chunks(self):
        """Ensure long text is properly split based on token limits."""
        sample_text = "This is a test sentence. " * 5000  # Simulate a long text
        chunks = split_text_into_chunks(sample_text)
        self.assertTrue(all(len(tiktoken.get_encoding("cl100k_base").encode(chunk)) <= MAX_TOKENS_PER_CHUNK for chunk in chunks) # Check token count for each chunk
)

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
