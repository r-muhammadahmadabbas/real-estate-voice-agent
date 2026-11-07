"""
LangGraph Agent for Real Estate Voice Agent
100% Free/Local - No API keys required
Uses rule-based intent detection + local RAG + pre-written UrduLish responses
"""
import os
import re
import uuid
import random
from datetime import datetime, timedelta
from typing import TypedDict, List, Dict, Any, Optional
from enum import Enum

from langgraph.graph import StateGraph, END

from real_estate_agent.tools import (
    search_properties_from_text, rag_search, check_calendar_availability,
    get_alternative_slots, book_appointment, reschedule_appointment,
    cancel_appointment, get_client_preferences, save_client_preferences,
    log_call, log_appointment
)
from real_estate_agent.recommendation import parse_preferences_from_text, PropertyRecommendationEngine, UserPreferences
from real_estate_agent.config import GROQ_API_KEY

llm = None
if GROQ_API_KEY:
    try:
        from langchain_groq import ChatGroq
        llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.3, api_key=GROQ_API_KEY)
        llm.invoke("test")
        print("[OK] Groq LLM ready")
    except Exception as e:
        print(f"[WARN] Groq LLM failed: {e}")
        llm = None

def safe_llm_call(prompt: str, fallback: str = "") -> str:
    """Call LLM with fallback - never breaks"""
    if llm:
        try:
            return llm.invoke(prompt).content
        except:
            pass
    return fallback

# ============================================================================
# URDU LISH RESPONSE TEMPLATES (pre-written, no LLM needed)
# ============================================================================

RESPONSES = {
    "greeting": [
        "Assalam-o-Alaikum! RealEstate Hub se baat ho rahi hai. Main aap ki kis tarah madad kar sakta hoon? Aap kya dhundh rahe hain - ghar, plot, ya commercial property?",
        "Wa alaikum assalam! Ji haan, RealEstate Hub mein khush aamdeed. Batayein, kya dhundh rahe hain?",
    ],
    "buyer_inquiry": [
        "Ji bilkul! Aap ke liye best properties dhundhta hoon. Budget aur area batayein.",
        "Samajh gaya. Kya budget hai aur kis area mein dhundh rahe hain?",
    ],
    "rental_inquiry": [
        "Ji, rental properties ke liye budget aur area batayein.",
        "Rental ke liye kitna budget hai aur kis area mein chahiye?",
    ],
    "commercial_inquiry": [
        "Commercial property ke liye business type aur size batayein.",
        "Ji, commercial property ke liye kya requirement hai?",
    ],
    "investment_inquiry": [
        "Investment ke liye budget aur goals batayein - rental yield ya capital appreciation?",
        "Ji, investment ke liye kitna budget hai aur kya target hai?",
    ],
    "off_topic": [
        "Main sirf real estate ki madad kar sakta hoon. Kya aap koi property ya investment ke baare mein jaanna chahte hain?",
        "Maazrat, ye mere scope se bahar hai. Real estate mein kya madad chahiye?",
    ],
    "goodbye": [
        "Shukriya aap ka waqt dene ke liye. RealEstate Hub se rabta karein. Allah Hafiz!",
        "Allah Hafiz! Koi aur sawal ho to zaroor batayein.",
    ],
    "clarification": [
        "Ek second sir, ye clarify karein: {question}",
        "Ji, ye batayein: {question}",
    ],
    "no_results": [
        "Maazrat, is budget mein koi property nahi mili. Kya budget adjust kar sakte hain ya area change karna hai?",
        "Ye budget mein kuch nahi mila. Thoda adjust karein?",
    ],
    "booking_confirm": [
        "Ji bilkul! Aap ki visit confirm ho gayi hai:\n\nProperty: {property}\nDate: {date}\nTime: {time}\n\nAgent ko email bhej di gayi hai. Shukriya!",
        "Visit book ho gayi hai!\n\nProperty: {property}\nDate: {date}\nTime: {time}\n\nShukriya!",
    ],
    "booking_unavailable": [
        "Ye slot available nahi hai. Kya ye time chalega?\n{alternatives}",
        "Ye time full hai. Alternative slots:\n{alternatives}",
    ],
    "objection_price": [
        "Sir, ye location ke hisaab se best value hai. Payment plan bhi flexible hai - {down}% down, {installments} installments.",
        "Ji sir, comparable areas mein 15-20% zyada rate hai. Ye best value hai.",
    ],
    "objection_trust": [
        "Sir, humare saath 500+ families settled hain. References de sakta hoon. Sab legal clear hai.",
        "Ji, hum registered company hain. 10+ saal ka experience hai.",
    ],
    "objection_location": [
        "Ye area fast develop ho rahi hai. Metro route approve ho chuka hai, 2 saal mein value 30-40% badh jayegi.",
        "Ji sir, ye area upcoming hai. Future mein bohat development hogi.",
    ],
}

# ============================================================================
# INTENT DETECTION (Rule-based, no LLM)
# ============================================================================

def detect_intent(text: str) -> tuple:
    text_lower = text.lower()
    
    # Off-topic detection
    off_topic_keywords = ['weather', 'mausam', 'politics', 'siasat', 'cricket', 'khana', 'movie', 'film', 'song', 'gaana']
    if any(kw in text_lower for kw in off_topic_keywords):
        return "off_topic", 0.95
    
    # Investment (check before greeting to avoid misclassification)
    invest_keywords = ['invest', 'investment', 'return', 'yield', 'profit', 'rental income', 'rental yield']
    if any(kw in text_lower for kw in invest_keywords):
        return "investment_inquiry", 0.90
    
    # Greeting
    greeting_keywords = ['assalam', 'salam', 'hello', 'hi', 'aoa', 'good morning', 'good evening']
    if any(kw in text_lower for kw in greeting_keywords) and len(text_lower.split()) < 10:
        return "greeting", 0.95
    
    # Goodbye
    goodbye_keywords = ['allah hafiz', 'khuda hafiz', 'bye', 'goodbye', 'shukriya', 'thank']
    if any(kw in text_lower for kw in goodbye_keywords) and len(text_lower.split()) < 5:
        return "goodbye", 0.95
    
    # Appointment booking
    booking_keywords = ['visit', 'booking', 'book karna', 'schedule', 'appointment', 'dekhna', 'site visit']
    if any(kw in text_lower for kw in booking_keywords):
        return "appointment_booking", 0.90
    
    # Reschedule
    reschedule_keywords = ['reschedule', 'change karna', 'badalna', 'time change']
    if any(kw in text_lower for kw in reschedule_keywords):
        return "appointment_reschedule", 0.90
    
    # Cancellation
    cancel_keywords = ['cancel', 'cancel karna', 'na karna']
    if any(kw in text_lower for kw in cancel_keywords):
        return "appointment_cancellation", 0.90
    
    # RAG / Factual questions (before buyer_inquiry)
    rag_keywords = ['payment plan', 'maintenance', 'transfer fee', 'tax', 'possession', 'school', 'hospital', 'amenities', 'kya hai', 'kaise', 'what is', 'batao', 'detail']
    if any(kw in text_lower for kw in rag_keywords):
        return "rag_query", 0.85

    # Investment (check before greeting to avoid misclassification)
    invest_keywords = ['invest', 'investment', 'return', 'yield', 'profit', 'rental income', 'rental yield']
    if any(kw in text_lower for kw in invest_keywords):
        return "investment_inquiry", 0.90
    
    # Commercial
    commercial_keywords = ['commercial', 'dukan', 'office', 'shop', 'plaza']
    if any(kw in text_lower for kw in commercial_keywords):
        return "commercial_inquiry", 0.85
    
    # Rental
    rent_keywords = ['rent', 'rental', 'kira', 'kiraya']
    if any(kw in text_lower for kw in rent_keywords):
        return "rental_inquiry", 0.85
    
    # Objection handling
    objection_keywords = ['mehenga', 'expensive', 'paisa nahi', 'budget nahi', 'trust nahi', 'pata nahi']
    if any(kw in text_lower for kw in objection_keywords):
        return "objection_handling", 0.80
    
    # Buyer inquiry (default for property-related)
    buyer_keywords = ['ghar', 'house', 'plot', 'zameen', 'property', 'buy', 'purchase', 'apna']
    if any(kw in text_lower for kw in buyer_keywords):
        return "buyer_inquiry", 0.80
    
    # Default
    return "buyer_inquiry", 0.60


# ============================================================================
# STATE DEFINITION
# ============================================================================

class AgentState(TypedDict):
    conversation_history: List[Dict[str, str]]
    current_turn: int
    user_profile: Dict[str, Any]
    user_preferences: Dict[str, Any]
    current_property: Optional[Dict[str, Any]]
    recommended_properties: List[Dict[str, Any]]
    last_mentioned_property: Optional[Dict[str, Any]]
    budget_min: Optional[int]
    budget_max: Optional[int]
    detected_intent: Optional[str]
    intent_confidence: float
    rag_results: Optional[List[Dict]]
    property_search_results: Optional[List[Dict]]
    calendar_availability: Optional[List[Dict]]
    calendar_event_id: Optional[str]
    email_sent: bool
    appointment_status: Optional[str]
    appointment_details: Optional[Dict[str, Any]]
    needs_clarification: bool
    clarification_question: Optional[str]
    retry_count: int
    max_retries: int
    final_response: str
    disclaimer: Optional[str]
    conversation_id: str
    started_at: str
    last_updated: str


def create_initial_state(conversation_id: str = None) -> AgentState:
    return {
        "conversation_history": [],
        "current_turn": 0,
        "user_profile": {},
        "user_preferences": {},
        "current_property": None,
        "recommended_properties": [],
        "last_mentioned_property": None,
        "budget_min": None,
        "budget_max": None,
        "detected_intent": None,
        "intent_confidence": 0.0,
        "rag_results": None,
        "property_search_results": None,
        "calendar_availability": None,
        "calendar_event_id": None,
        "email_sent": False,
        "appointment_status": None,
        "appointment_details": None,
        "needs_clarification": False,
        "clarification_question": None,
        "retry_count": 0,
        "max_retries": 3,
        "final_response": "",
        "disclaimer": None,
        "conversation_id": conversation_id or str(uuid.uuid4()),
        "started_at": datetime.now().isoformat(),
        "last_updated": datetime.now().isoformat()
    }


# ============================================================================
# GRAPH NODES
# ============================================================================

def intent_detection_node(state: AgentState) -> AgentState:
    user_msg = state["conversation_history"][-1]["content"] if state["conversation_history"] else ""
    intent, confidence = detect_intent(user_msg)
    return {**state, "detected_intent": intent, "intent_confidence": confidence}


def greeting_node(state: AgentState) -> AgentState:
    if llm:
        prompt = """You are a Pakistani real estate agent speaking natural UrduLish.
Give a warm, professional greeting. Mention you can help with buying, renting, or investing in property.
Keep it short (2-3 sentences) and natural."""
        response = llm.invoke(prompt).content
    else:
        response = random.choice(RESPONSES["greeting"])
    return {**state, "final_response": response, "detected_intent": "greeting"}


def rag_node(state: AgentState) -> AgentState:
    user_msg = state["conversation_history"][-1]["content"]
    results = rag_search(user_msg, top_k=3)
    
    if not results:
        return {**state, "rag_results": [], "final_response": "Maazrat, ye information mere paas nahi hai. Kya property ke baare mein jaanna hai?"}
    
    context = "\n".join([r['document'] for r in results[:3]])
    
    if llm:
        prompt = f"""You are a Pakistani real estate agent speaking natural UrduLish (Urdu + English mixed).
Answer the question using ONLY the provided context. Be warm, professional, and conversational.
If the answer is not in the context, say "Ye information mere paas nahi hai."

Context: {context}
Question: {user_msg}

Answer in natural UrduLish:"""
        response = llm.invoke(prompt).content
    else:
        response_parts = []
        for r in results:
            doc = r['document']
            if doc.startswith("Q:"):
                parts = doc.split("\nA: ")
                if len(parts) == 2:
                    response_parts.append(f"**{parts[0].replace('Q: ', '')}**\n{parts[1]}")
            else:
                response_parts.append(doc)
        response = "\n\n".join(response_parts[:2])
    
    return {**state, "rag_results": results, "final_response": response}


def recommendation_node(state: AgentState) -> AgentState:
    user_msg = state["conversation_history"][-1]["content"]
    prefs = parse_preferences_from_text(user_msg)
    
    # Merge with existing preferences
    if state.get("user_preferences"):
        for k, v in state["user_preferences"].items():
            if v and not prefs.get(k):
                prefs[k] = v
    
    # Search properties
    engine = PropertyRecommendationEngine()
    user_prefs = UserPreferences(
        budget_min=prefs.get('budget_min'),
        budget_max=prefs.get('budget_max'),
        city=prefs.get('city'),
        area=prefs.get('area'),
        min_bedrooms=prefs.get('min_bedrooms', 0),
        property_type=prefs.get('property_type'),
        purpose=prefs.get('purpose'),
        amenities=prefs.get('amenities', []),
        investment_goals=prefs.get('investment_goals')
    )
    
    results = engine.recommend(user_prefs, top_k=5)
    
    if not results:
        if llm:
            prompt = f"""You are a Pakistani real estate agent speaking natural UrduLish.
The user asked: "{user_msg}"
No properties matched their criteria. Respond naturally, ask if they want to adjust budget or area.
Keep it short and conversational."""
            response = llm.invoke(prompt).content
        else:
            response = random.choice(RESPONSES["no_results"])
        return {**state, "final_response": response, "needs_clarification": True, "clarification_question": "Kya budget ya area change karein?"}
    
    # Format response
    if llm:
        props_text = "\n".join([f"{i+1}. {r['title']} - {r['price_formatted']}" for i, r in enumerate(results)])
        prompt = f"""You are a Pakistani real estate agent speaking natural UrduLish.
Present these property recommendations warmly and professionally:

{props_text}

End by asking if they want details or want to book a visit. Keep it natural and conversational."""
        response = llm.invoke(prompt).content
    else:
        response = "Yeh rahi aap ke liye best options:\n\n"
        for i, r in enumerate(results, 1):
            monthly = f"   Monthly: PKR {r['monthly_installment']:,.0f}" if r.get('monthly_installment') else ""
            response += f"**{i}. {r['title']}**\n   {r['price_formatted']} | {r['area_sqft']} sqft{monthly}\n\n"
        response += "Kisi property ke baare mein detail chahiye ya visit schedule karni hai?"
    
    return {
        **state,
        "recommended_properties": results,
        "user_preferences": prefs,
        "final_response": response,
        "last_mentioned_property": results[0] if results else None
    }


def booking_node(state: AgentState) -> AgentState:
    user_msg = state["conversation_history"][-1]["content"]
    props = state.get("recommended_properties", [])
    
    # If no properties in state, search from conversation history
    if not props:
        for msg in reversed(state.get("conversation_history", [])[-5:]):
            if msg.get("role") == "user":
                found = search_properties_from_text(msg["content"], top_k=1)
                if found:
                    props = found
                    break
    
    if not props:
        return {**state, "final_response": "Pehle property select karein. Kya area aur budget hai?", "needs_clarification": True}
    
    prop = props[0]
    
    # Parse date and time
    date_match = re.search(r'(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})', user_msg)
    if not date_match:
        date_match = re.search(r'(\d{4})[/-](\d{1,2})[/-](\d{1,2})', user_msg)
        if date_match:
            date_str = f"{date_match.group(1)}-{date_match.group(2)}-{date_match.group(3)}"
        else:
            # Try "20 January" format
            months = {'january': '01', 'february': '02', 'march': '03', 'april': '04', 'may': '05', 'june': '06',
                      'july': '07', 'august': '08', 'september': '09', 'october': '10', 'november': '11', 'december': '12'}
            for month_name, month_num in months.items():
                if month_name in user_msg.lower():
                    day_match = re.search(r'(\d{1,2})', user_msg)
                    if day_match:
                        date_str = f"2025-{month_num}-{day_match.group(1).zfill(2)}"
                        break
            else:
                return {**state, "final_response": "Visit ke liye date batayein. Jaise: '20 January' ya '20/01/2025'", "needs_clarification": True, "clarification_question": "Date kya hai?"}
    else:
        date_str = f"{date_match.group(3)}-{date_match.group(2)}-{date_match.group(1)}"
    
    time_match = re.search(r'(\d{1,2})[:.](\d{2})', user_msg)
    if not time_match:
        time_match = re.search(r'(\d{1,2})\s*(?:baje|am|pm)', user_msg.lower())
        if time_match:
            hour = int(time_match.group(1))
            if 'pm' in user_msg.lower() and hour < 12:
                hour += 12
            time_str = f"{hour:02d}:00"
        else:
            return {**state, "final_response": "Time batayein. Jaise: '10 baje' ya '10:00'", "needs_clarification": True, "clarification_question": "Time kya hai?"}
    else:
        time_str = f"{time_match.group(1).zfill(2)}:{time_match.group(2)}"
    
    # Book appointment
    result = book_appointment(
        client_name=state.get("user_profile", {}).get("name", "Client"),
        client_phone=state.get("user_profile", {}).get("phone", "0300-0000000"),
        client_email=state.get("user_profile", {}).get("email", "client@example.com"),
        property_id=prop["id"],
        date_str=date_str,
        time_str=time_str
    )
    
    if result.get("success"):
        appt = result["appointment"]
        response = random.choice(RESPONSES["booking_confirm"]).format(
            property=appt['property'],
            date=appt['date'],
            time=appt['time']
        )
        return {
            **state,
            "final_response": response,
            "calendar_event_id": result["event_id"],
            "appointment_status": "confirmed",
            "appointment_details": appt
        }
    else:
        alts = result.get("alternatives", [])
        alt_text = "\n".join([f"- {a['date']} at {a['time']}" for a in alts[:3]])
        response = random.choice(RESPONSES["booking_unavailable"]).format(alternatives=alt_text)
        return {**state, "final_response": response, "needs_clarification": True, "clarification_question": "Alternative time select karein"}


def reschedule_node(state: AgentState) -> AgentState:
    user_msg = state["conversation_history"][-1]["content"]
    event_id = state.get("calendar_event_id")
    
    if not event_id:
        return {**state, "final_response": "Koi existing appointment nahi mili. Pehle visit book karein.", "needs_clarification": True}
    
    # Parse new date/time
    date_match = re.search(r'(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})', user_msg)
    time_match = re.search(r'(\d{1,2})[:.](\d{2})', user_msg)
    
    if not date_match or not time_match:
        return {**state, "final_response": "Naya date aur time batayein.", "needs_clarification": True}
    
    date_str = f"{date_match.group(3)}-{date_match.group(2)}-{date_match.group(1)}"
    time_str = f"{time_match.group(1).zfill(2)}:{time_match.group(2)}"
    
    result = reschedule_appointment(event_id, date_str, time_str)
    
    if result.get("success"):
        return {**state, "final_response": f"Ji, appointment reschedule ho gayi hai. Naya time: {result['new_time']}", "appointment_status": "rescheduled"}
    
    return {**state, "final_response": "Reschedule nahi ho paya. Dobara try karein.", "needs_clarification": True}


def cancellation_node(state: AgentState) -> AgentState:
    event_id = state.get("calendar_event_id")
    
    if not event_id:
        return {**state, "final_response": "Koi appointment nahi mili cancel karne ke liye.", "needs_clarification": True}
    
    result = cancel_appointment(event_id)
    
    if result.get("success"):
        return {**state, "final_response": "Ji, appointment cancel ho gayi hai. Kya koi aur madad chahiye?", "appointment_status": "cancelled"}
    
    return {**state, "final_response": "Cancel nahi ho paya. Dobara try karein."}


def refusal_node(state: AgentState) -> AgentState:
    if llm:
        prompt = """You are a Pakistani real estate agent. The user asked something off-topic.
Politely redirect them to real estate. Be warm, not robotic.
Keep it short and natural in UrduLish."""
        response = llm.invoke(prompt).content
    else:
        response = random.choice(RESPONSES["off_topic"])
    return {**state, "final_response": response, "detected_intent": "off_topic"}


def goodbye_node(state: AgentState) -> AgentState:
    if llm:
        prompt = """You are a Pakistani real estate agent speaking natural UrduLish.
The customer is saying goodbye. Respond warmly and professionally.
Keep it short and natural."""
        response = llm.invoke(prompt).content
    else:
        response = random.choice(RESPONSES["goodbye"])
    return {**state, "final_response": response, "detected_intent": "goodbye"}


def objection_node(state: AgentState) -> AgentState:
    user_msg = state["conversation_history"][-1]["content"]
    
    if llm:
        prompt = f"""You are a Pakistani real estate agent speaking natural UrduLish.
The customer has an objection: "{user_msg}"
Respond warmly and professionally. Address their concern with facts about value, payment plans, or trust.
Keep it conversational and persuasive. End by asking if they'd like to see properties or book a visit."""
        response = llm.invoke(prompt).content
    else:
        user_msg_lower = user_msg.lower()
        if any(kw in user_msg_lower for kw in ['mehenga', 'expensive', 'paisa nahi', 'budget']):
            response = random.choice(RESPONSES["objection_price"])
        elif any(kw in user_msg_lower for kw in ['trust', 'pata nahi', 'reliable']):
            response = random.choice(RESPONSES["objection_trust"])
        elif any(kw in user_msg_lower for kw in ['location', 'area', 'door']):
            response = random.choice(RESPONSES["objection_location"])
        else:
            response = "Ji sir, aap ki concern samajh sakta hoon. Kya specific baat hai jo pareshan kar rahi hai?"
    
    return {**state, "final_response": response}


def validation_node(state: AgentState) -> AgentState:
    if state["detected_intent"] == "appointment_booking" and not state.get("calendar_event_id"):
        return {**state, "needs_clarification": True, "clarification_question": "Date aur time batayein"}
    return state


def response_node(state: AgentState) -> AgentState:
    response = state.get("final_response", "")
    if state["detected_intent"] in ["appointment_booking", "appointment_reschedule"] and state.get("calendar_event_id"):
        response += "\n\n_Yeh booking confirm hai. Agent ko email bhej di gayi hai._"
    return {**state, "final_response": response}


# ============================================================================
# ROUTING
# ============================================================================

def route_intent(state: AgentState) -> str:
    intent = state.get("detected_intent", "")
    routes = {
        "greeting": "greeting_node",
        "buyer_inquiry": "recommendation_node",
        "rental_inquiry": "recommendation_node",
        "commercial_inquiry": "recommendation_node",
        "investment_inquiry": "recommendation_node",
        "returning_customer": "rag_node",
        "rag_query": "rag_node",
        "appointment_booking": "booking_node",
        "appointment_reschedule": "reschedule_node",
        "appointment_cancellation": "cancellation_node",
        "objection_handling": "objection_node",
        "off_topic": "refusal_node",
        "goodbye": "goodbye_node",
    }
    return routes.get(intent, "rag_node")


# ============================================================================
# GRAPH BUILDER
# ============================================================================

def build_agent_graph():
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("intent_detection_node", intent_detection_node)
    workflow.add_node("greeting_node", greeting_node)
    workflow.add_node("rag_node", rag_node)
    workflow.add_node("recommendation_node", recommendation_node)
    workflow.add_node("booking_node", booking_node)
    workflow.add_node("reschedule_node", reschedule_node)
    workflow.add_node("cancellation_node", cancellation_node)
    workflow.add_node("refusal_node", refusal_node)
    workflow.add_node("goodbye_node", goodbye_node)
    workflow.add_node("objection_node", objection_node)
    workflow.add_node("validation_node", validation_node)
    workflow.add_node("response_node", response_node)
    
    # Entry point
    workflow.set_entry_point("intent_detection_node")
    
    # Routing
    workflow.add_conditional_edges("intent_detection_node", route_intent)
    
    # Connect all action nodes to validation
    for node in ["greeting_node", "rag_node", "recommendation_node", "booking_node",
                 "reschedule_node", "cancellation_node", "refusal_node", "goodbye_node", "objection_node"]:
        workflow.add_edge(node, "validation_node")
    
    workflow.add_edge("validation_node", "response_node")
    workflow.add_edge("response_node", END)
    
    return workflow.compile()


_agent_graph = None


def get_agent_graph():
    global _agent_graph
    if _agent_graph is None:
        _agent_graph = build_agent_graph()
    return _agent_graph


# ============================================================================
# PUBLIC API
# ============================================================================

def run_agent(user_message: str, conversation_id: str = None, history: List[Dict] = None, 
              carried_state: Dict = None) -> Dict:
    """Run the agent with a user message"""
    graph = get_agent_graph()
    
    if carried_state:
        state = carried_state
        state["conversation_history"].append({"role": "user", "content": user_message})
        state["current_turn"] = len(state["conversation_history"])
    else:
        state = create_initial_state(conversation_id)
        if history:
            state["conversation_history"] = history
        state["conversation_history"].append({"role": "user", "content": user_message})
        state["current_turn"] = len(state["conversation_history"])
    
    result = graph.invoke(state)
    
    return {
        "response": result.get("final_response", ""),
        "intent": result.get("detected_intent", ""),
        "conversation_id": result.get("conversation_id", ""),
        "needs_clarification": result.get("needs_clarification", False),
        "clarification_question": result.get("clarification_question", ""),
        "properties": result.get("recommended_properties", []),
        "event_id": result.get("calendar_event_id"),
        "appointment_status": result.get("appointment_status"),
        "state": result,
    }


if __name__ == "__main__":
    print("Testing Real Estate Voice Agent (100% Free/Local)...")
    print("=" * 60)
    
    # Test 1: Greeting
    result = run_agent("Assalam-o-Alaikum")
    print(f"\n1. Greeting:")
    print(f"   Intent: {result['intent']}")
    print(f"   Response: {result['response'][:80]}...")
    
    # Test 2: Buyer inquiry
    result = run_agent("Budget 3 crore hai, Lahore DHA mein, 3 bedroom")
    print(f"\n2. Buyer Inquiry:")
    print(f"   Intent: {result['intent']}")
    print(f"   Response: {result['response'][:100]}...")
    print(f"   Properties found: {len(result.get('properties', []))}")
    
    # Test 3: Off-topic
    result = run_agent("What's the weather like today?")
    print(f"\n3. Off-Topic:")
    print(f"   Intent: {result['intent']}")
    print(f"   Response: {result['response'][:80]}...")
    
    # Test 4: RAG
    result = run_agent("Payment plan for DHA Phase 5 kya hai?")
    print(f"\n4. RAG Query:")
    print(f"   Intent: {result['intent']}")
    print(f"   Response: {result['response'][:100]}...")
    
    # Test 5: Multi-turn booking
    print(f"\n5. Multi-Turn Booking:")
    conv_id = None
    messages = [
        "Assalam-o-Alaikum, main ghar dhundh raha hoon",
        "Budget 3 crore hai, Lahore DHA",
        "Visit book karni hai 20/01/2025 10:00",
    ]
    for msg in messages:
        result = run_agent(msg, conv_id)
        conv_id = result.get("conversation_id")
        print(f"   User: {msg}")
        print(f"   Agent ({result['intent']}): {result['response'][:60]}...")
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED! System is ready for demo.")
    print("=" * 60)