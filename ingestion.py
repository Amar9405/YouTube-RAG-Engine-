from dotenv import load_dotenv
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

load_dotenv()

def extract_video_id(url_or_id: str) -> str:
    """Extracts the 11-character video ID from any YouTube URL or raw ID."""
    url_or_id = url_or_id.strip()
    
    # If it's a URL
    if "youtube.com" in url_or_id or "youtu.be" in url_or_id:
        parsed_url = urlparse(url_or_id)
        
        # youtu.be/xxxxx
        if parsed_url.hostname == "youtu.be":
            return parsed_url.path.lstrip("/")
            
        # youtube.com/watch?v=xxxxx
        if "youtube.com" in parsed_url.hostname:
            if parsed_url.path == "/watch":
                return parse_qs(parsed_url.query).get("v", [""])[0]
            
            # youtube.com/shorts/xxxxx
            if "/shorts/" in parsed_url.path:
                return parsed_url.path.split("/shorts/")[1].split("/")[0]
                
    # Assume it's already a raw ID
    return url_or_id

def get_transcript(video_id: str) -> str:
    """Fetches transcript, trying English first, then Hindi."""
    video_id = extract_video_id(video_id)
    
    try:
        transcript_list = YouTubeTranscriptApi().fetch(video_id, languages=['en', 'hi'])
        return " ".join(snippet.text for snippet in transcript_list)
    except TranscriptsDisabled:
        raise Exception("Captions are disabled for this video.")
    except NoTranscriptFound:
        raise Exception("No English or Hindi transcript found for this video.")
    except Exception:
        raise Exception("Video unavailable. It may be deleted, private, or the URL is invalid.")

def split_transcript(transcript: str, video_id: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    doc = Document(
        page_content=transcript,
        metadata={"video_id": video_id, "source": "youtube"},
    )
    return splitter.split_documents([doc])