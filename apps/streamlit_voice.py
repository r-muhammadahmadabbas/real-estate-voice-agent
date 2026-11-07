"""
Robust Voice Agent UI - Never breaks, always works
Run: streamlit run voice_ui.py
"""
import streamlit as st
import sys
import os
import base64
import tempfile
import logging
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from real_estate_agent.agent import run_agent
from real_estate_agent.voice_pipeline import text_to_speech, transcribe_audio, is_whisper_available

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="RealEstate Hub - Voice Agent", page_icon="🏠", layout="wide")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None
if "carried_state" not in st.session_state:
    st.session_state.carried_state = None

# Header
st.title("🏠 RealEstate Hub - AI Voice Agent")
st.markdown("### Real Estate Voice Assistant | UrduLish")

# Sidebar
with st.sidebar:
    st.header("Status")
    
    # Show system status
    whisper_ok = is_whisper_available()
    st.success("✅ Text-to-Speech: Ready") if st.button("Test TTS") else None
    if whisper_ok:
        st.success("✅ Speech-to-Text: Ready")
    else:
        st.warning("⚠️ Speech-to-Text: Not available (text input only)")
    
    st.markdown("---")
    st.header("Quick Actions")
    if st.button("🏠 Buy Property"):
        st.session_state.messages.append({"role": "user", "content": "Main ghar dhundh raha hoon, budget 3 crore"})
        st.rerun()
    if st.button("🔑 Rent Property"):
        st.session_state.messages.append({"role": "user", "content": "Main rental property dhundh raha hai"})
        st.rerun()
    if st.button("📈 Investment"):
        st.session_state.messages.append({"role": "user", "content": "Main investment karna chahta hoon"})
        st.rerun()
    if st.button("📅 Book Visit"):
        st.session_state.messages.append({"role": "user", "content": "Main property visit book karna chahta hoon"})
        st.rerun()
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.conversation_id = None
        st.session_state.carried_state = None
        st.rerun()
    
    st.markdown("---")
    st.info("💡 Type in Urdu, Roman Urdu, or English!")

# Display chat history
for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            if st.button("🔊", key=f"play_{idx}"):
                try:
                    with st.spinner("Generating audio..."):
                        audio_bytes = text_to_speech(message["content"])
                        if audio_bytes:
                            b64 = base64.b64encode(audio_bytes).decode()
                            audio_html = f'<audio controls autoplay style="width:100%;height:40px;"><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>'
                            st.markdown(audio_html, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Audio error: {e}")

# Voice Output Section (always visible)
if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
    st.markdown("---")
    st.markdown("### 🔊 Agent Voice Response:")
    try:
        audio_bytes = text_to_speech(st.session_state.messages[-1]["content"])
        if audio_bytes:
            b64 = base64.b64encode(audio_bytes).decode()
            audio_html = f'<audio controls autoplay style="width:100%;height:50px;"><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>'
            st.markdown(audio_html, unsafe_allow_html=True)
    except Exception as e:
        logger.error(f"TTS error: {e}")

# Text input (always works)
if prompt := st.chat_input("Apna sawal yahan likhein..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Soch raha hoon..."):
                result = run_agent(
                    prompt,
                    st.session_state.conversation_id,
                    carried_state=st.session_state.carried_state
                )
                st.session_state.conversation_id = result.get("conversation_id")
                st.session_state.carried_state = result.get("state")
                response = result.get("response", "Maazrat, samajh nahi aaya.")
                st.markdown(response)
                
                # Auto-play audio prominently
                try:
                    with st.spinner("Audio ban raha hai..."):
                        audio_bytes = text_to_speech(response)
                        if audio_bytes:
                            st.markdown("### 🔊 Agent Response:")
                            b64 = base64.b64encode(audio_bytes).decode()
                            audio_html = f'<audio controls autoplay style="width:100%;height:40px;"><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>'
                            st.markdown(audio_html, unsafe_allow_html=True)
                except Exception as e:
                    logger.error(f"TTS error: {e}")
                
                # Show properties
                if result.get("properties"):
                    st.subheader("🏘️ Recommended Properties")
                    for p in result["properties"]:
                        with st.expander(f"{p['title']} - {p['price_formatted']}"):
                            st.write(f"**Type:** {p['type']}")
                            st.write(f"**Area:** {p['area_sqft']} sqft")
                            if p.get('monthly_installment'):
                                st.write(f"**Monthly:** PKR {p['monthly_installment']:,.0f}")
                
                if result.get("needs_clarification"):
                    st.warning(result.get("clarification_question", ""))
        except Exception as e:
            logger.error(f"Agent error: {e}")
            st.error(f"Error: {e}. Please try again.")
    
    st.session_state.messages.append({"role": "assistant", "content": response})

# Voice input section
st.markdown("---")
st.subheader("🎤 Voice Input")

# Show voice status
if not whisper_ok:
    st.info("Voice input requires Whisper. Text input works perfectly!")
else:
    st.success("Voice input ready! Record below or upload audio.")

# Audio recorder
try:
    audio_value = st.audio_input("Record your voice")
    if audio_value is not None:
        with st.spinner("Audio process ho raha hai..."):
            try:
                # Transcribe
                transcription = transcribe_audio(audio_value.getvalue())
                
                if transcription:
                    st.success(f"🎤 You said: {transcription}")
                    
                    # Process through agent
                    with st.spinner("Agent respond kar raha hai..."):
                        result = run_agent(
                            transcription,
                            st.session_state.conversation_id,
                            carried_state=st.session_state.carried_state
                        )
                        st.session_state.conversation_id = result.get("conversation_id")
                        st.session_state.carried_state = result.get("state")
                        response = result.get("response", "")
                        
                        st.markdown(f"**🤖 Agent:** {response}")
                        
                        # Play audio response
                        try:
                            audio_resp = text_to_speech(response)
                            if audio_resp:
                                b64 = base64.b64encode(audio_resp).decode()
                                audio_html = f'<audio controls autoplay><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>'
                                st.markdown(audio_html, unsafe_allow_html=True)
                        except Exception as e:
                            logger.error(f"TTS error: {e}")
                        
                        # Save to history
                        st.session_state.messages.append({"role": "user", "content": transcription})
                        st.session_state.messages.append({"role": "assistant", "content": response})
                else:
                    st.warning("Audio samajh nahi aaya. Text input use karein.")
            except Exception as e:
                logger.error(f"Voice processing error: {e}")
                st.error(f"Voice error: {e}. Text input use karein.")
except Exception as e:
    logger.error(f"Audio input error: {e}")
    st.info("Voice recording not available in this browser. Use text input or file upload.")

# File upload option
st.markdown("---")
st.subheader("📁 Upload Audio File")
audio_file = st.file_uploader("Upload WAV/MP3 file", type=["wav", "mp3"])
if audio_file is not None:
    with st.spinner("Processing..."):
        try:
            audio_bytes = audio_file.read()
            transcription = transcribe_audio(audio_bytes)
            
            if transcription:
                st.success(f"Transcribed: {transcription}")
                
                with st.spinner("Agent respond kar raha hai..."):
                    result = run_agent(
                        transcription,
                        st.session_state.conversation_id,
                        carried_state=st.session_state.carried_state
                    )
                    st.session_state.conversation_id = result.get("conversation_id")
                    st.session_state.carried_state = result.get("state")
                    response = result.get("response", "")
                    
                    st.markdown(f"**Agent:** {response}")
                    
                    try:
                        audio_resp = text_to_speech(response)
                        if audio_resp:
                            b64 = base64.b64encode(audio_resp).decode()
                            audio_html = f'<audio controls autoplay><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>'
                            st.markdown(audio_html, unsafe_allow_html=True)
                    except Exception as e:
                        logger.error(f"TTS error: {e}")
                    
                    st.session_state.messages.append({"role": "user", "content": transcription})
                    st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                st.warning("Audio samajh nahi aaya.")
        except Exception as e:
            logger.error(f"File processing error: {e}")
            st.error(f"Error: {e}")

st.markdown("---")
st.caption("RealEstate Hub AI Voice Agent | Whisper STT + gTTS + LangGraph + RAG")