"""
System Health Check
===================
Checks every major component (agent, RAG, recommendations, voice, tools and the
SQLite database) and prints a pass/fail summary.

Run from the repository root:

    python scripts/health_check.py
"""
import os
import sys

from real_estate_agent.config import DB_PATH

def check_component(name, check_func):
    try:
        result = check_func()
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} - {name}")
        return result
    except Exception as e:
        print(f"[ERROR] - {name}: {e}")
        return False

def main():
    print("\n" + "=" * 60)
    print("  SYSTEM HEALTH CHECK")
    print("=" * 60 + "\n")
    
    results = []
    
    # Check 1: Agent
    def check_agent():
        from real_estate_agent.agent import run_agent
        result = run_agent("Assalam-o-Alaikum")
        return result['intent'] == 'greeting'
    results.append(check_component("Agent (LangGraph)", check_agent))
    
    # Check 2: RAG
    def check_rag():
        from real_estate_agent.rag_pipeline import RealEstateRAG
        rag = RealEstateRAG()
        rag.load_data()
        rag.initialize_vector_store()
        results = rag.search("payment plan", top_k=1)
        return len(results) > 0
    results.append(check_component("RAG Pipeline", check_rag))
    
    # Check 3: Recommendation
    def check_recommendation():
        from real_estate_agent.recommendation import PropertyRecommendationEngine, UserPreferences
        engine = PropertyRecommendationEngine()
        prefs = UserPreferences(budget_max=30000000, city="Lahore")
        results = engine.recommend(prefs, top_k=1)
        return len(results) > 0
    results.append(check_component("Recommendation Engine", check_recommendation))
    
    # Check 4: TTS
    def check_tts():
        from real_estate_agent.voice_pipeline import text_to_speech
        audio = text_to_speech("Test")
        return len(audio) > 0
    results.append(check_component("Text-to-Speech (gTTS)", check_tts))
    
    # Check 5: STT
    def check_stt():
        from real_estate_agent.voice_pipeline import is_whisper_available
        return is_whisper_available()
    results.append(check_component("Speech-to-Text (Whisper)", check_stt))
    
    # Check 6: Tools
    def check_tools():
        from real_estate_agent.tools import check_calendar_availability
        result = check_calendar_availability("2025-01-20", "10:00")
        return result.get('available', False)
    results.append(check_component("Calendar Tools", check_tools))
    
    # Check 7: Database
    def check_db():
        from real_estate_agent.tools import init_db
        init_db()
        return os.path.exists(DB_PATH)
    results.append(check_component("Database (SQLite)", check_db))
    
    # Summary
    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"  RESULTS: {passed}/{total} passed")
    print("=" * 60)
    
    if passed == total:
        print("\n[OK] All systems ready! Run: streamlit run apps/streamlit_voice.py")
    else:
        print("\n⚠️ Some components need attention.")
        print("Text input will work regardless of STT status.")
    
    print()

if __name__ == "__main__":
    main()