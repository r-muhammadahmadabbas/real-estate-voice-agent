"""
Robust Voice Pipeline: Speech-to-Text + Text-to-Speech
Handles errors gracefully, never breaks the app
"""
import os
import io
import tempfile
import logging
import subprocess

from real_estate_agent.config import DEEPGRAM_API_KEY

logger = logging.getLogger(__name__)

# ============================================================================
# SPEECH TO TEXT (Whisper - free, local)
# ============================================================================

_whisper_model = None
_whisper_available = None

def is_whisper_available() -> bool:
    """Check if Whisper is available"""
    global _whisper_available
    if _whisper_available is None:
        try:
            import whisper
            _whisper_available = True
        except ImportError:
            _whisper_available = False
    return _whisper_available

def get_whisper_model():
    """Load Whisper model (cached)"""
    global _whisper_model
    if _whisper_model is None:
        import whisper
        logger.info("Loading Whisper model (base)...")
        _whisper_model = whisper.load_model("base")
        logger.info("Whisper model loaded")
    return _whisper_model

def convert_to_wav(input_path: str, output_path: str) -> bool:
    """Convert any audio format to WAV using ffmpeg"""
    try:
        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
            output_path
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        return result.returncode == 0
    except Exception as e:
        logger.error(f"FFmpeg conversion error: {e}")
        return False

def convert_audio_bytes(audio_data: bytes) -> bytes:
    """Convert audio bytes to WAV format using pydub"""
    try:
        from pydub import AudioSegment
        import io
        
        # Try to load as various formats
        for format_name in ['wav', 'mp3', 'webm', 'ogg']:
            try:
                audio = AudioSegment.from_file(io.BytesIO(audio_data), format=format_name)
                audio = audio.set_frame_rate(16000).set_channels(1)
                output = io.BytesIO()
                audio.export(output, format="wav")
                return output.getvalue()
            except:
                continue
        return audio_data
    except ImportError:
        return audio_data
    except Exception as e:
        logger.error(f"Audio conversion error: {e}")
        return audio_data

def transcribe_audio_deepgram(audio_data: bytes) -> str:
    """Transcribe audio using Deepgram"""
    if not DEEPGRAM_API_KEY:
        return ""
    
    try:
        import requests
        import io
        
        # Convert to WAV first
        wav_data = convert_audio_bytes(audio_data)
        
        response = requests.post(
            "https://api.deepgram.com/v1/listen?model=nova-3&language=ur&smart_format=true",
            headers={
                "Authorization": f"Token {DEEPGRAM_API_KEY}",
                "Content-Type": "audio/wav"
            },
            data=wav_data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            text = result.get('results', {}).get('channels', [{}])[0].get('alternatives', [{}])[0].get('transcript', '')
            return text.strip()
        else:
            logger.error(f"Deepgram error: {response.status_code} - {response.text[:200]}")
            return ""
            
    except Exception as e:
        logger.error(f"Deepgram transcription error: {e}")
        return ""

def transcribe_audio(audio_data: bytes) -> str:
    """Transcribe audio bytes to text - tries Deepgram first, falls back to Whisper"""
    # Try Deepgram first (better quality)
    if DEEPGRAM_API_KEY:
        text = transcribe_audio_deepgram(audio_data)
        if text:
            return text
    
    # Fall back to Whisper
    return transcribe_audio_whisper(audio_data)

def transcribe_audio_whisper(audio_data: bytes) -> str:
    """Transcribe audio bytes to text using Whisper - no ffmpeg needed"""
    if not is_whisper_available():
        logger.warning("Whisper not available")
        return ""
    
    try:
        import numpy as np
        from pydub import AudioSegment
        import io
        
        # Load audio with pydub
        audio = None
        for fmt in ['wav', 'mp3', 'webm', 'ogg', 'm4a']:
            try:
                audio = AudioSegment.from_file(io.BytesIO(audio_data), format=fmt)
                break
            except:
                continue
        
        if audio is None:
            return ""
        
        # Convert to 16kHz mono numpy array (Whisper's expected format)
        audio = audio.set_frame_rate(16000).set_channels(1)
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        samples = samples / 32768.0  # Normalize to [-1, 1]
        
        # Transcribe directly from numpy array (no file, no ffmpeg)
        model = get_whisper_model()
        result = model.transcribe(samples, language="ur", fp16=False)
        text = result["text"].strip()
        
        if text:
            logger.info(f"Transcribed: {text}")
        return text
        
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return ""

def transcribe_audio_file(file_path: str) -> str:
    """Transcribe audio file to text"""
    if not is_whisper_available():
        return ""
    
    temp_wav = None
    
    try:
        # Convert to WAV if needed
        if not file_path.endswith('.wav'):
            temp_wav = file_path + "_converted.wav"
            if not convert_to_wav(file_path, temp_wav):
                temp_wav = file_path
        else:
            temp_wav = file_path
        
        model = get_whisper_model()
        result = model.transcribe(temp_wav, language="ur", fp16=False)
        return result["text"].strip()
        
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return ""
        
    finally:
        if temp_wav and temp_wav != file_path and os.path.exists(temp_wav):
            try:
                os.unlink(temp_wav)
            except:
                pass

# ============================================================================
# TEXT TO SPEECH (gTTS - free, Google)
# ============================================================================

def text_to_speech(text: str, lang: str = "ur") -> bytes:
    """Convert text to speech audio bytes using gTTS"""
    if not text or not text.strip():
        return b""
    
    try:
        from gtts import gTTS
        
        # Clean text for TTS (remove markdown)
        clean_text = text.replace("**", "").replace("*", "").replace("\n", " ")
        
        tts = gTTS(text=clean_text, lang=lang, slow=False)
        
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        
        return audio_buffer.read()
        
    except Exception as e:
        logger.error(f"TTS error: {e}")
        return b""

def text_to_speech_file(text: str, output_path: str, lang: str = "ur") -> str:
    """Convert text to speech and save to file"""
    try:
        from gtts import gTTS
        
        clean_text = text.replace("**", "").replace("*", "").replace("\n", " ")
        tts = gTTS(text=clean_text, lang=lang, slow=False)
        tts.save(output_path)
        return output_path
        
    except Exception as e:
        logger.error(f"TTS error: {e}")
        return ""

# ============================================================================
# VOICE PIPELINE (Combined)
# ============================================================================

class VoicePipeline:
    """Complete voice pipeline: Audio → Text → Agent → Text → Audio"""
    
    def __init__(self):
        self.stt_available = is_whisper_available()
        if not self.stt_available:
            logger.warning("Whisper not available - voice input disabled")
    
    def process_voice(self, audio_data: bytes, agent_callback) -> dict:
        """Process voice input through the pipeline"""
        # Step 1: Transcribe
        transcription = transcribe_audio(audio_data)
        
        if not transcription:
            error_msg = "Audio samajh nahi aaya. Dobara try karein ya type karein."
            return {
                "transcription": "",
                "response_text": error_msg,
                "response_audio": text_to_speech(error_msg),
                "error": True
            }
        
        # Step 2: Get agent response
        response_text = agent_callback(transcription)
        
        # Step 3: Convert to speech
        response_audio = text_to_speech(response_text)
        
        return {
            "transcription": transcription,
            "response_text": response_text,
            "response_audio": response_audio,
            "error": False
        }
    
    def is_ready(self) -> bool:
        """Check if voice pipeline is ready"""
        return self.stt_available


if __name__ == "__main__":
    # Test TTS
    print("Testing TTS...")
    audio = text_to_speech("Assalam-o-Alaikum! RealEstate Hub mein khush aamdeed.")
    print(f"Generated {len(audio)} bytes of audio")
    
    with open("test_output.mp3", "wb") as f:
        f.write(audio)
    print("Saved to test_output.mp3")
    
    # Check STT
    print(f"\nWhisper available: {is_whisper_available()}")
    if is_whisper_available():
        print("STT ready!")
    else:
        print("Install whisper: pip install openai-whisper")