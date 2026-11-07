"""
FastAPI Backend for Real Estate Voice Agent
Run: uvicorn api:app --reload
"""
import os
import sys
import uuid
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from real_estate_agent.agent import run_agent
from real_estate_agent.voice_pipeline import text_to_speech, transcribe_audio
from real_estate_agent.rag_pipeline import RealEstateRAG
from real_estate_agent.recommendation import PropertyRecommendationEngine, UserPreferences

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RealEstate Hub - AI Voice Agent API",
    description="Production-ready real estate voice agent with RAG, recommendations, and appointment booking",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# MODELS
# ============================================================================

class ChatRequest(BaseModel):
    message: str = Field(..., description="User message")
    conversation_id: Optional[str] = Field(None, description="Conversation ID for multi-turn")

class ChatResponse(BaseModel):
    response: str
    intent: str
    conversation_id: str
    needs_clarification: bool = False
    clarification_question: Optional[str] = None
    properties: List[Dict] = []
    event_id: Optional[str] = None
    appointment_status: Optional[str] = None

class PropertySearchRequest(BaseModel):
    query: str = Field(..., description="Natural language property search")
    budget_max: Optional[int] = None
    city: Optional[str] = None
    area: Optional[str] = None
    min_bedrooms: Optional[int] = None
    top_k: int = 5

class RAGQueryRequest(BaseModel):
    query: str = Field(..., description="Question to search knowledge base")
    top_k: int = 5

class BookingRequest(BaseModel):
    client_name: str
    client_phone: str
    client_email: str
    property_id: str
    date: str = Field(..., description="Date in YYYY-MM-DD format")
    time: str = Field(..., description="Time in HH:MM format")
    notes: str = ""

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str

# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/", response_model=HealthResponse)
async def root():
    """API info"""
    return {
        "name": "RealEstate Hub - AI Voice Agent API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": ["/chat", "/search", "/rag", "/book", "/health"]
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint - send message, get agent response"""
    try:
        result = run_agent(
            request.message,
            conversation_id=request.conversation_id
        )
        return ChatResponse(
            response=result["response"],
            intent=result["intent"],
            conversation_id=result["conversation_id"],
            needs_clarification=result.get("needs_clarification", False),
            clarification_question=result.get("clarification_question"),
            properties=result.get("properties", []),
            event_id=result.get("event_id"),
            appointment_status=result.get("appointment_status")
        )
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search", response_model=List[Dict])
async def search_properties(request: PropertySearchRequest):
    """Search properties using natural language"""
    try:
        from real_estate_agent.tools import search_properties_from_text
        results = search_properties_from_text(
            request.query,
            budget_max=request.budget_max,
            city=request.city,
            area=request.area,
            min_bedrooms=request.min_bedrooms,
            top_k=request.top_k
        )
        return results
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rag", response_model=List[Dict])
async def rag_query(request: RAGQueryRequest):
    """Search knowledge base for factual answers"""
    try:
        rag = RealEstateRAG()
        rag.load_data()
        rag.initialize_vector_store()
        results = rag.search(request.query, top_k=request.top_k)
        return results
    except Exception as e:
        logger.error(f"RAG error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/book")
async def book_appointment(request: BookingRequest):
    """Book a property visit appointment"""
    try:
        from real_estate_agent.tools import book_appointment as book_appt
        result = book_appt(
            client_name=request.client_name,
            client_phone=request.client_phone,
            client_email=request.client_email,
            property_id=request.property_id,
            date_str=request.date,
            time_str=request.time,
            notes=request.notes
        )
        return result
    except Exception as e:
        logger.error(f"Booking error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/voice/tts")
async def text_to_speech_endpoint(text: str):
    """Convert text to speech"""
    try:
        audio_bytes = text_to_speech(text)
        from fastapi.responses import Response
        return Response(content=audio_bytes, media_type="audio/mpeg")
    except Exception as e:
        logger.error(f"TTS error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/voice/stt")
async def speech_to_text(audio: bytes):
    """Convert speech to text"""
    try:
        text = transcribe_audio(audio)
        return {"transcription": text}
    except Exception as e:
        logger.error(f"STT error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)