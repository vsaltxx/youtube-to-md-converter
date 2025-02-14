import unittest
from unittest.mock import patch, MagicMock, mock_open, call
import os
import httpx

from youtube_to_md import (
    get_video_metadata,
    safe_filename,
    download_audio,
    split_audio,
    detect_dominant_language,
    transcribe_audio,
    format_transcript_with_llama,
    save_transcript_to_markdown
)

from youtube_to_md import TOO_MANY_REQUESTS_ERROR

class TestYouTubeToMarkdown(unittest.TestCase):
    """Unit tests for YouTube to Markdown conversion functions."""

    @patch("yt_dlp.YoutubeDL")
    def test_get_video_metadata_valid_url(self, mock_ytdlp):
        """Test fetching video metadata from a valid YouTube URL."""
        mock_instance = MagicMock()
        mock_ytdlp.return_value.__enter__.return_value = mock_instance
        mock_instance.extract_info.return_value = {
            "title": "Test Video",
            "duration": 300,
            "uploader": "Test Uploader",
            "id": "abcd1234"
        }

        result = get_video_metadata("https://www.youtube.com/watch?v=example")
        self.assertEqual(result["title"], "Test Video")
        self.assertEqual(result["duration"], 300)
        self.assertEqual(result["uploader"], "Test Uploader")
        self.assertEqual(result["video_id"], "abcd1234")

    @patch("yt_dlp.YoutubeDL")
    def test_get_video_metadata_missing_fields(self, mock_ytdlp):
        """Test fetching video metadata when some fields are missing."""
        mock_instance = MagicMock()
        mock_ytdlp.return_value.__enter__.return_value = mock_instance
        mock_instance.extract_info.return_value = {}

        result = get_video_metadata("https://www.youtube.com/watch?v=example")
        self.assertEqual(result["title"], "Unknown Title")
        self.assertEqual(result["duration"], 0)
        self.assertEqual(result["uploader"], "Unknown")
        self.assertEqual(result["video_id"], "unknown")

    def test_safe_filename(self):
        """Test that invalid characters are removed from filenames."""
        title = 'Test: Video/Title*?<>|'
        video_id = 'abcd1234'
        expected = 'Test_VideoTitle_abcd1234'
        result = safe_filename(title, video_id)
        self.assertEqual(result, expected)

    def test_safe_filename_long_title(self):
        """Test that long filenames are truncated properly."""
        title = "a" * 200 # 100 characters is the limit for the title
        video_id = "abcd1234"
        result = safe_filename(title, video_id)
        self.assertLessEqual(len(result), 109)  # 100 for title, 1 for '_', 8 for 'abcd1234'

    def test_safe_filename_unicode(self):
        """Test handling of non-ASCII characters in filenames."""
        title = "测试视频标题"  # Chinese characters
        video_id = "abcd1234"
        result = safe_filename(title, video_id)
        self.assertIn("测试视频标题", result)

    @patch("yt_dlp.YoutubeDL")
    @patch("os.rename")
    @patch("os.path.exists", return_value=True)
    @patch("pydub.AudioSegment.from_file")
    @patch("os.remove")
    def test_download_audio_valid_url(self, mock_remove, mock_pydub, mock_exists, mock_rename, mock_ytdlp):
        """Test successful download and conversion of YouTube audio."""
        mock_instance = MagicMock()
        mock_ytdlp.return_value.__enter__.return_value = mock_instance
        mock_instance.extract_info.return_value = {
            "title": "Sample Video",
            "id": "abcd1234",
        }
        mock_instance.prepare_filename.return_value = "downloads/Sample_Video.m4a"

        valid_url = "https://www.youtube.com/watch?v=DFYRQ_zQ-gk"
        metadata = get_video_metadata(valid_url)
        result = download_audio(valid_url, metadata)

        expected_output = os.path.normpath("downloads/Sample_Video_abcd1234.m4a")
        self.assertEqual(os.path.normpath(result), expected_output)

    @patch("yt_dlp.YoutubeDL")
    def test_download_audio_invalid_url(self, mock_ytdlp):
        """Test that an invalid YouTube URL returns None."""
        mock_instance = MagicMock()
        mock_ytdlp.return_value.__enter__.return_value = mock_instance

        from yt_dlp.utils import DownloadError
        mock_instance.extract_info.side_effect = DownloadError("Mocked DownloadError")

        invalid_url = "https://invalid.url"
        metadata = {"title": "Invalid Video", "video_id": "invalid123"}
        result = download_audio(invalid_url, metadata)
        self.assertIsNone(result)

    @patch("pydub.AudioSegment.from_file")
    @patch("os.path.exists", return_value=True)
    def test_split_audio(self, mock_isfile, mock_pydub):
        """Test splitting audio into chunks."""
        mock_audio = MagicMock()
        mock_audio.__len__.return_value = 26 * 60 * 1000 # 26 minutes in milliseconds (3 chunks)
        mock_pydub.return_value = mock_audio

        result = split_audio("test_audio.mp3")
        self.assertEqual(len(result), 3)

    @patch("pydub.AudioSegment.from_file")
    @patch("os.path.exists", return_value=True)
    def test_split_audio_short_audio(self, mock_isfile, mock_pydub):
        """Test splitting an audio file that is shorter than the chunk size."""
        mock_audio = MagicMock()
        mock_audio.__len__.return_value = 5 * 60 * 1000  # 5 minutes
        mock_pydub.return_value = mock_audio
        result = split_audio("short_audio.mp3")
        self.assertEqual(len(result), 1)

    def test_detect_dominant_language(self):
        """Test detecting dominant language in text."""
        text = "Hello world!\nBonjour tout le monde!\nHello again!"
        filtered_text, lang = detect_dominant_language(text)
        self.assertEqual(lang, "en")

    def test_detect_dominant_language_empty_text(self):
        """Test detecting language when text is empty."""
        filtered_text, lang = detect_dominant_language("")
        self.assertEqual(lang, "unknown")

    @patch("os.remove")
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
        self.assertEqual(transcript,
                         "Mocked transcription result\nMocked transcription result\nMocked transcription result")

    @patch("youtube_to_md.groq.Client")
    def test_transcribe_audio_rate_limit(self, mock_groq_client):
        """Test handling of API rate limiting during transcription."""
        mock_client_instance = mock_groq_client.return_value
        mock_client_instance.audio.transcriptions.create.side_effect = httpx.HTTPStatusError(
            "Rate limit exceeded",
            request=MagicMock(),
            response=MagicMock(status_code=TOO_MANY_REQUESTS_ERROR)
        )
        transcript, detected_language = transcribe_audio("dummy_audio.mp3")
        self.assertEqual(transcript, "")
        self.assertEqual(detected_language, "unknown")

    @patch("youtube_to_md.groq.Client")
    def test_format_transcript_with_llama(self, mock_groq_client):
        """Test processing text with LLaMA API to format the transcript."""
        mock_client_instance = MagicMock()
        mock_groq_client.return_value = mock_client_instance
        mock_client_instance.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="## Formatted Markdown"))]
        )
        result = format_transcript_with_llama("Sample transcript")
        self.assertTrue(result.startswith("## Formatted Markdown"))

    @patch("youtube_to_md.groq.Client")
    def test_format_transcript_with_llama_empty_text(self, mock_groq_client):
        """Test processing an empty transcript with LLaMA API."""
        mock_client_instance = MagicMock()
        mock_groq_client.return_value = mock_client_instance
        mock_client_instance.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=""))]
        )
        result = format_transcript_with_llama("")
        self.assertEqual(result, "")

    @patch("os.makedirs")
    @patch("builtins.open", new_callable=mock_open)
    def test_save_transcript_to_markdown(self, mock_open_file, mock_makedirs):
        """Test saving the transcript as a Markdown file."""
        metadata = {
            "title": "Test Video",
            "uploader": "Test Uploader",
            "duration": 300,
            "video_id": "abcd1234"
        }
        save_transcript_to_markdown("Formatted text", metadata, "https://youtube.com/watch?v=abcd123")

        mock_open_file.assert_called_once_with(
            os.path.join("articles", "Test_Video_abcd1234.md"), "w", encoding="utf-8"
        )

    @patch("os.makedirs")
    @patch("builtins.open", new_callable=mock_open)
    def test_save_transcript_to_markdown_missing_metadata(self, mock_open_file, mock_makedirs):
        """Test saving the transcript when metadata fields are missing, ensuring missing keys are handled safely."""
        metadata = {
            "title": "Unknown Title",
            "video_id": "unknown",
            "uploader": "Unknown",
            "duration": 0}
        save_transcript_to_markdown("Formatted text", metadata, "https://youtube.com/watch?v=abcd123")
        mock_open_file.assert_called() # File should still be created

    @patch("yt_dlp.YoutubeDL")
    @patch("os.path.exists", return_value=True)
    @patch("os.remove")
    @patch("os.rename")
    @patch("pydub.AudioSegment.from_file")
    def test_file_with_downloaded_file_name_exists(self, mock_pydub, mock_rename, mock_remove, mock_exists, mock_ytdlp):
        """Test handling when the file with the downloaded file name already exists."""
        mock_instance = MagicMock()
        mock_ytdlp.return_value.__enter__.return_value = mock_instance
        mock_instance.extract_info.return_value = {
            "title": "Sample Video",
            "id": "abcd1234",
        }
        mock_instance.prepare_filename.return_value = os.path.normpath("downloads/Sample_Video.m4a")

        valid_url = "https://www.youtube.com/watch?v=DFYRQ_zQ-gk"
        metadata = get_video_metadata(valid_url)
        result = download_audio(valid_url, metadata)

        expected_output = os.path.normpath("downloads/Sample_Video_abcd1234.m4a")

        # Assert file removal was called due to pre-existing file
        mock_remove.assert_any_call(os.path.normpath("downloads/Sample_Video_abcd1234.m4a"))

        # Ensure rename was still called
        mock_rename.assert_called_once_with(
            os.path.normpath("downloads/Sample_Video.m4a"),
            os.path.normpath("downloads/Sample_Video_abcd1234.m4a")
        )

        self.assertEqual(os.path.normpath(result), expected_output)


if __name__ == "__main__":
    unittest.main()
