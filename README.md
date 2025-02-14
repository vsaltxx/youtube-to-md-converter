# YouTube to Markdown Converter

## Overview
This Python script automates the process of converting a YouTube video into a readable Markdown article. It downloads 
audio from the video, transcribes the speech into text, formats it using a language model, and saves the final article 
as a Markdown file.

## Features
- ✅ **Downloads YouTube audio** using yt-dlp.
- ✅ **Transcribes speech** using OpenAI Whisper (via Groq API).
- ✅ **Formats text into Markdown** using LLaMA 3.
- ✅ **Automatically cleans up** temporary files.
- ✅ **Provides real-time feedback** with a progress bar for transcription.
- ✅ **Handles errors gracefully** with retry mechanisms for API rate limits.

## Requirements
### **1️⃣ Install Dependencies**
Before using the script, install all required dependencies:
```sh
pip install -r requirements.txt
```

### **2️⃣ Set Up API Keys**
This script uses the **Groq API** for transcription and formatting. Create a `.env` file and add your API key:
```sh
GROQ_API_KEY=your_groq_api_key_here
```

## Installation & Usage
### **Run the script**
```sh
python youtube_to_md.py "<YouTube Video URL>"
```

Example:
```sh
python youtube_to_md.py "https://www.youtube.com/watch?v=DFYRQ_zQ-gk"
```

### **Output**
The script generates:
- **A Markdown file** with the transcribed and formatted content in `articles/`.
- **Automatic cleanup** of the `downloads/` folder after completion.

## Example Output
```md
# Video Title

**Uploader:** Example Uploader

**Duration:** 600 seconds

**Original Video:** [Video Title](https://www.youtube.com/watch?v=DFYRQ_zQ-gk)

---

## Introduction
This is the transcribed and formatted content...
```

## Running Tests
### **Unit Tests**
This project includes automated unit tests to ensure reliability.

#### **Run all tests**
```sh
python -m unittest discover tests
```

#### **Run a specific test file**
```sh
python -m unittest tests.test_youtube_to_md
```

#### **Run a specific test case**
```sh
python -m unittest tests.test_youtube_to_md.TestYouTubeToMarkdown.test_transcribe_audio_rate_limit
```

### **What’s Tested?**
- ✅ Video metadata extraction
- ✅ Audio download and format conversion
- ✅ Speech-to-text transcription
- ✅ Markdown formatting
- ✅ Error handling (e.g., missing API key, failed downloads, API rate limits)

## Limitations
| **Limitation**                     | **Details** |
|--------------------------------------|-------------|
| **Language Support** | Transcription accuracy depends on Whisper’s model capabilities. Some languages may have lower accuracy. |
| **Audio Length** | Whisper API handles audio in segments. Long videos take longer to process. |
| **YouTube Restrictions** | Some videos may be unavailable for download due to region restrictions, copyright, or age limits. |
| **Rate Limits** | Groq API has a limit on the number of transcription requests. The script implements exponential backoff, but prolonged use may still be restricted. |
| **Formatting Accuracy** | LLaMA formatting is AI-based and may not always produce perfectly structured Markdown. |
| **18+ and Private Videos** | The script cannot process videos that are age-restricted (18+) or set as private on YouTube. |

## Error Handling & Debugging
| **Issue**                        | **Solution** |
|----------------------------------|-------------|
| `Error: GROQ API key is missing.` | Ensure you have set `GROQ_API_KEY` in `.env` |
| `Download failed.` | Try a VPN or check if the video is restricted. |
| `Rate limit exceeded.` | The script will retry automatically. |
| `Low disk space.` | Ensure at least **500MB** of free storage. |
## Error Handling & Debugging
| **Issue**                        | **Solution** |
|----------------------------------|-------------|
| `Error: GROQ API key is missing.` | Ensure you have set `GROQ_API_KEY` in `.env` |
| `Download failed.` | Try a VPN or check if the video is restricted. |
| `Rate limit exceeded.` | The script will retry automatically. |
| `Low disk space.` | Ensure at least **500MB** of free storage. |

## References & Links
Here are the primary resources and dependencies used in this project:
- [yt-dlp Documentation](https://github.com/yt-dlp/yt-dlp)
- [OpenAI Whisper](https://github.com/openai/whisper)
- [Groq API](https://groq.com/)
- [Python Unittest](https://docs.python.org/3/library/unittest.html)
 
## License
This project is licensed under the GNU General Public License v3.0.

