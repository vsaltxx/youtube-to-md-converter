import unittest
from unittest.mock import patch, MagicMock, mock_open, call
import os
from youtube_to_md import (
    get_video_metadata,
    safe_filename,
    download_audio_from_youtube,
    split_audio,
    detect_dominant_language,
    transcribe_audio,
    process_text_with_llama,
    format_transcript_to_markdown
)


class TestYouTubeToMarkdown(unittest.TestCase):

    ## ============================= ##
    ##  1. TEST YOUTUBE METADATA ##
    ## ============================= ##

    @patch("yt_dlp.YoutubeDL")
    def test_get_video_metadata_valid_url(self, mock_ytdlp):
        """Test fetching video metadata from a valid YouTube URL."""
        mock_instance = MagicMock()
        mock_ytdlp.return_value.__enter__.return_value = mock_instance
        mock_instance.extract_info.return_value = {
            "title": "Test Video",
            "duration": 300,
            "uploader": "Test Uploader",
        }

        result = get_video_metadata("https://www.youtube.com/watch?v=example")

        self.assertEqual(result["title"], "Test Video")
        self.assertEqual(result["duration"], 300)
        self.assertEqual(result["uploader"], "Test Uploader")

    @patch("yt_dlp.YoutubeDL")
    def test_get_video_metadata_invalid_url(self, mock_ytdlp):
        """Test fetching metadata for an invalid URL returns None."""
        mock_instance = MagicMock()
        mock_ytdlp.return_value.__enter__.return_value = mock_instance

        # Simulate extract_info raising yt_dlp.utils.DownloadError
        from yt_dlp.utils import DownloadError
        mock_instance.extract_info.side_effect = DownloadError("Mocked DownloadError")

        result = get_video_metadata("invalid_url")

        # The function should handle the exception and return None
        self.assertIsNone(result)

        # Ensure extract_info was called once with the invalid URL
        mock_instance.extract_info.assert_called_once_with("invalid_url", download=False)

    ## ============================== ##
    ##  2. TEST SAFE FILENAME ##
    ## ============================== ##

    def test_safe_filename(self):
        """Test that invalid characters are removed from filenames."""
        title = 'Test: Video/Title*?<>|'
        video_id = 'abcd1234'
        expected = 'Test_VideoTitle_abcd1234'
        result = safe_filename(title, video_id)
        self.assertEqual(result, expected)

    ## ======================================== ##
    ##  3. TEST AUDIO DOWNLOAD & CONVERSION ##
    ## ======================================== ##

    @patch("yt_dlp.YoutubeDL")  # Mock yt_dlp
    @patch("os.rename")  # Mock os.rename
    @patch("os.path.exists", return_value=True)  # Simulate file existence
    @patch("pydub.AudioSegment.from_file")  # Mock Pydub
    @patch("os.remove")  # Mock os.remove
    def test_download_audio_valid_url(self, mock_remove, mock_pydub, mock_exists, mock_rename, mock_ytdlp):
        """Test successful download and conversion of YouTube audio."""
        mock_instance = MagicMock()
        mock_ytdlp.return_value.__enter__.return_value = mock_instance
        mock_instance.extract_info.return_value = {
            "title": "Sample Video",
            "id": "abcd1234",
        }
        mock_instance.prepare_filename.return_value = "downloads/Sample_Video.m4a"

        mock_audio_segment = MagicMock()
        mock_pydub.return_value = mock_audio_segment
        mock_audio_segment.export.return_value = None

        valid_url = "https://www.youtube.com/watch?v=DFYRQ_zQ-gk"
        result = download_audio_from_youtube(valid_url)

        expected_output = os.path.normpath("downloads/Sample_Video_abcd1234.mp3")
        self.assertEqual(os.path.normpath(result), expected_output)

    @patch("yt_dlp.YoutubeDL")
    def test_download_audio_invalid_url(self, mock_ytdlp):
        """Test that an invalid YouTube URL returns None."""
        mock_instance = MagicMock()
        mock_ytdlp.return_value.__enter__.return_value = mock_instance

        from yt_dlp.utils import DownloadError
        mock_instance.extract_info.side_effect = DownloadError("Mocked DownloadError")

        invalid_url = "https://invalid.url"
        result = download_audio_from_youtube(invalid_url)
        self.assertIsNone(result)

    ## =============================== ##
    ##  4. TEST AUDIO SPLITTING ##
    ## =============================== ##

    @patch("pydub.AudioSegment.from_file")
    @patch("os.path.exists", return_value=True)
    def test_split_audio(self, mock_isfile, mock_pydub):
        """Test splitting audio into chunks."""
        mock_audio = MagicMock()
        mock_audio.__len__.return_value = 26 * 60 * 1000  # 26 minutes (>10 min) in milliseconds: 26 / 10 = 3 chunks
        mock_pydub.return_value = mock_audio

        result = split_audio("test_audio.mp3")
        self.assertEqual(len(result), 3)  # Ensure three chunks are created

    ## =============================== ##
    ##  5. TEST LANGUAGE DETECTION ##
    ## =============================== ##

    def test_detect_dominant_language(self):
        """Test detecting dominant language in text."""
        text = "Hello world!\nBonjour tout le monde!\nHello again!"
        filtered_text, lang = detect_dominant_language(text)
        self.assertEqual(lang, "en")  # English should be dominant

    ## =============================== ##
    ##  6. TEST AUDIO TRANSCRIPTION ##
    ## =============================== ##

    @patch("os.remove")  # Mock os.remove to prevent FileNotFoundError
    @patch("youtube_to_md.split_audio", return_value=["chunk1.mp3", "chunk2.mp3", "chunk3.mp3"])
    @patch("youtube_to_md.groq.Client")
    @patch("builtins.open", new_callable=mock_open, read_data=b"dummy data")
    @patch("os.path.isfile", return_value=True)
    def test_transcribe_audio_success(self, mock_isfile, mock_file, mock_groq_client, mock_split_audio, mock_remove):
        """Test transcribing audio successfully using mocked API."""
        mock_response = MagicMock()
        mock_response.text = "Mocked transcription result"
        mock_response.language = "en"

        mock_client_instance = mock_groq_client.return_value
        mock_client_instance.audio.transcriptions.create.return_value = mock_response

        transcript, detected_language = transcribe_audio("dummy_audio.mp3")

        self.assertEqual(transcript,"Mocked transcription result\nMocked transcription result\nMocked transcription result")

        # Ensure chunk files were removed
        mock_remove.assert_any_call("chunk1.mp3")
        mock_remove.assert_any_call("chunk2.mp3")
        mock_remove.assert_any_call("chunk3.mp3")

    ## ================================= ##
    ##  7. TEST TEXT FORMATTING (LLAMA) ##
    ## ================================= ##

    @patch("youtube_to_md.groq.Client")
    def test_process_text_with_llama(self, mock_groq_client):
        """
        Test processing text with LLaMA API.
        Ensures only the correctly formatted text is returned (without extra messages
        like "Here is the formatted text:").
        """

        mock_client_instance = MagicMock()
        mock_groq_client.return_value = mock_client_instance
        mock_client_instance.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="## Formatted Markdown"))]
        )

        result = process_text_with_llama("Sample transcript", "en")
        self.assertTrue(result.startswith("## Formatted Markdown"))

    ## =============================== ##
    ##  8. TEST MARKDOWN OUTPUT ##
    ## =============================== ##

    @patch("os.makedirs")
    @patch("builtins.open", new_callable=mock_open)
    def test_format_transcript_to_markdown(self, mock_open_file, mock_makedirs):
        """Test saving the transcript as a Markdown file."""
        metadata = {
            "title": "Test Video",
            "uploader": "Test Uploader",
            "duration": 300,
        }
        format_transcript_to_markdown("Formatted text", metadata, "https://youtube.com/watch?v=abcd123")

        mock_open_file.assert_called_once_with(
            os.path.join("articles", "Test_Video_abcd123.md"), "w", encoding="utf-8"
        )

    @patch("yt_dlp.YoutubeDL")
    @patch("os.path.exists", return_value=True)  # Simulate that the expected file already exists
    @patch("os.remove")  # Mock os.remove to prevent actual file deletion
    @patch("os.rename")  # Mock os.rename to avoid real renaming
    @patch("pydub.AudioSegment.from_file")  # Mock Pydub
    def test_file_with_downloaded_file_name_exists(self, mock_pudub, mock_rename, mock_remove, mock_exists, mock_ytdlp):
        """Test handling when the file with the downloaded file name already exists."""
        mock_instance = MagicMock()
        mock_ytdlp.return_value.__enter__.return_value = mock_instance
        mock_instance.extract_info.return_value = {
            "title": "Sample Video",
            "id": "abcd1234",
        }
        mock_instance.prepare_filename.return_value = os.path.normpath("downloads/Sample_Video.m4a")

        valid_url = "https://www.youtube.com/watch?v=DFYRQ_zQ-gk"
        result = download_audio_from_youtube(valid_url)

        expected_output = os.path.normpath("downloads/Sample_Video_abcd1234.mp3")

        # Assert file removal was called due to pre-existing file
        mock_remove.assert_has_calls([
            call(os.path.normpath("downloads/Sample_Video_abcd1234.m4a")),
            call(os.path.normpath("downloads/Sample_Video_abcd1234.m4a"))
        ], any_order=False)

        # Ensure rename was still called
        mock_rename.assert_called_once_with(
            os.path.normpath("downloads/Sample_Video.m4a"),
            os.path.normpath("downloads/Sample_Video_abcd1234.m4a")
        )

        self.assertEqual(os.path.normpath(result), expected_output)

if __name__ == "__main__":
    unittest.main()
