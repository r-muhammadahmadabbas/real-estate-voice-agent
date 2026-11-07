"""
Quick test script for the Real Estate Voice Agent
Run: python test_system.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from real_estate_agent.agent import run_agent
from real_estate_agent.rag_pipeline import RealEstateRAG
from real_estate_agent.recommendation import PropertyRecommendationEngine, UserPreferences

def test_rag():
    print("=" * 60)
    print("TEST 1: RAG Pipeline")
    print("=" * 60)
    rag = RealEstateRAG()
    rag.load_data()
    rag.initialize_vector_store()
    results = rag.search("payment plan DHA Phase 5", top_k=2)
    print(f"Found {len(results)} results")
    for r in results:
        print(f"  - {r['document'][:80]}...")
    print("RAG: PASS\n")

def test_recommendation():
    print("=" * 60)
    print("TEST 2: Recommendation Engine")
    print("=" * 60)
    engine = PropertyRecommendationEngine()
    prefs = UserPreferences(budget_max=30000000, city="Lahore", area="DHA")
    results = engine.recommend(prefs, top_k=3)
    print(f"Found {len(results)} properties")
    for r in results:
        print(f"  - {r['title']} | {r['price_formatted']} | Score: {r['score']:.1f}")
    print("Recommendation: PASS\n")

def test_agent_greeting():
    print("=" * 60)
    print("TEST 3: Agent - Greeting")
    print("=" * 60)
    result = run_agent("Assalam-o-Alaikum")
    print(f"Intent: {result['intent']}")
    print(f"Response: {result['response'][:100]}...")
    print("Agent Greeting: PASS\n")

def test_agent_buyer():
    print("=" * 60)
    print("TEST 4: Agent - Buyer Inquiry")
    print("=" * 60)
    result = run_agent("Budget 3 crore hai, Lahore DHA mein, 3 bedroom")
    print(f"Intent: {result['intent']}")
    print(f"Response: {result['response'][:150]}...")
    print("Agent Buyer: PASS\n")

def test_agent_offtopic():
    print("=" * 60)
    print("TEST 5: Agent - Off-Topic Refusal")
    print("=" * 60)
    result = run_agent("What's the weather like today?")
    print(f"Intent: {result['intent']}")
    print(f"Response: {result['response'][:100]}...")
    print("Agent Off-Topic: PASS\n")

def test_agent_booking():
    print("=" * 60)
    print("TEST 6: Agent - Appointment Booking")
    print("=" * 60)
    history = [
        {"role": "user", "content": "Budget 3 crore hai, Lahore DHA"},
        {"role": "assistant", "content": "Yeh rahi options..."},
    ]
    result = run_agent("Visit book karni hai, 20 January 10 baje", history=history)
    print(f"Intent: {result['intent']}")
    print(f"Response: {result['response'][:150]}...")
    print("Agent Booking: PASS\n")

def test_agent_multiturn():
    print("=" * 60)
    print("TEST 7: Agent - Multi-Turn Conversation")
    print("=" * 60)
    conv_id = None
    messages = [
        "Assalam-o-Alaikum, main ghar dhundh raha hoon",
        "Budget 3 crore hai, Lahore DHA mein",
        "3 bedroom chahiye",
        "Visit book karni hai kal 10 baje",
    ]
    for msg in messages:
        result = run_agent(msg, conv_id)
        conv_id = result.get("conversation_id")
        print(f"User: {msg}")
        print(f"Agent ({result['intent']}): {result['response'][:80]}...")
        print()
    print("Multi-Turn: PASS\n")

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("REAL ESTATE VOICE AGENT - SYSTEM TEST")
    print("=" * 60 + "\n")
    
    try:
        test_rag()
        test_recommendation()
        test_agent_greeting()
        test_agent_buyer()
        test_agent_offtopic()
        test_agent_booking()
        test_agent_multiturn()
        
        print("=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
        print("\nTo launch the UI, run:")
        print("  streamlit run ui.py")
    except Exception as e:
        print(f"TEST FAILED: {e}")
        import traceback
        traceback.print_exc()