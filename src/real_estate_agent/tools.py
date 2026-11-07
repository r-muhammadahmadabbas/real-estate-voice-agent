"""
Tools for the Real Estate Voice Agent
"""
import os
import json
import smtplib
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import sqlite3
import logging

from real_estate_agent.config import (
    AGENT_EMAIL,
    DB_PATH,
    SMTP_HOST,
    SMTP_PASSWORD,
    SMTP_PORT,
    SMTP_USER,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# DATABASE SETUP
# ============================================================================

DB_PATH = str(DB_PATH)

def init_db():
    """Initialize SQLite database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Call logs
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS call_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            call_id TEXT UNIQUE,
            client_phone TEXT,
            client_name TEXT,
            start_time TIMESTAMP,
            end_time TIMESTAMP,
            duration_seconds INTEGER,
            transcript TEXT,
            intent TEXT,
            outcome TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Client preferences
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS client_preferences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            client_phone TEXT UNIQUE,
            client_name TEXT,
            budget_min INTEGER,
            budget_max INTEGER,
            preferred_cities TEXT,
            preferred_areas TEXT,
            min_bedrooms INTEGER,
            property_type TEXT,
            preferred_amenities TEXT,
            investment_goals TEXT,
            last_contact TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Appointments
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS appointments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            call_log_id INTEGER,
            client_phone TEXT,
            property_id TEXT,
            agent_email TEXT,
            calendar_event_id TEXT,
            scheduled_time TIMESTAMP,
            status TEXT DEFAULT 'scheduled',
            cancellation_reason TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (call_log_id) REFERENCES call_logs (id)
        )
    """)
    
    # Follow-ups
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS follow_ups (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            client_phone TEXT,
            appointment_id INTEGER,
            reminder_type TEXT,
            due_date TIMESTAMP,
            status TEXT DEFAULT 'pending',
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (appointment_id) REFERENCES appointments (id)
        )
    """)
    
    conn.commit()
    conn.close()

# Initialize on import
init_db()

# ============================================================================
# CRM TOOLS
# ============================================================================

def get_client_preferences(client_phone: str) -> Optional[Dict]:
    """Get client preferences from CRM"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM client_preferences WHERE client_phone = ?", (client_phone,))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return dict(row)
    return None

def save_client_preferences(client_phone: str, client_name: str, preferences: Dict) -> bool:
    """Save or update client preferences"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO client_preferences 
        (client_phone, client_name, budget_min, budget_max, preferred_cities, 
         preferred_areas, min_bedrooms, property_type, preferred_amenities, 
         investment_goals, last_contact)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(client_phone) DO UPDATE SET
            client_name = excluded.client_name,
            budget_min = excluded.budget_min,
            budget_max = excluded.budget_max,
            preferred_cities = excluded.preferred_cities,
            preferred_areas = excluded.preferred_areas,
            min_bedrooms = excluded.min_bedrooms,
            property_type = excluded.property_type,
            preferred_amenities = excluded.preferred_amenities,
            investment_goals = excluded.investment_goals,
            last_contact = excluded.last_contact,
            updated_at = CURRENT_TIMESTAMP
    """, (
        client_phone,
        client_name,
        preferences.get('budget_min'),
        preferences.get('budget_max'),
        json.dumps(preferences.get('preferred_cities', [])),
        json.dumps(preferences.get('preferred_areas', [])),
        preferences.get('min_bedrooms', 0),
        preferences.get('property_type'),
        json.dumps(preferences.get('preferred_amenities', [])),
        preferences.get('investment_goals'),
        datetime.now().isoformat()
    ))
    
    conn.commit()
    conn.close()
    return True

def log_call(call_id: str, client_phone: str, client_name: str, 
             transcript: str, intent: str, outcome: str) -> int:
    """Log call to CRM"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO call_logs (call_id, client_phone, client_name, start_time, 
                               transcript, intent, outcome)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (call_id, client_phone, client_name, datetime.now().isoformat(), 
          transcript, intent, outcome))
    
    log_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return log_id

def log_appointment(call_log_id: int, client_phone: str, property_id: str, 
                    agent_email: str, calendar_event_id: str, 
                    scheduled_time: datetime) -> int:
    """Log appointment to CRM"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO appointments 
        (call_log_id, client_phone, property_id, agent_email, calendar_event_id, scheduled_time)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (call_log_id, client_phone, property_id, agent_email, 
          calendar_event_id, scheduled_time.isoformat()))
    
    appt_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return appt_id

# ============================================================================
# CALENDAR TOOLS (Google Calendar API)
# ============================================================================

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

def _get_calendar_service():
    """Get Google Calendar API service"""
    try:
        from googleapiclient.discovery import build
        return build('calendar', 'v3', developerKey=GOOGLE_API_KEY)
    except ImportError:
        return None

# Mock calendar storage (fallback when API not available)
MOCK_CALENDAR = {}

def check_calendar_availability(date_str: str, time_str: str, duration_minutes: int = 60) -> Dict:
    """Check if a time slot is available"""
    try:
        dt = datetime.fromisoformat(f"{date_str}T{time_str}")
    except:
        for fmt in ["%Y-%m-%d %H:%M", "%d/%m/%Y %H:%M", "%Y-%m-%dT%H:%M"]:
            try:
                dt = datetime.strptime(f"{date_str} {time_str}" if "T" not in date_str else f"{date_str}T{time_str}", fmt)
                break
            except:
                continue
        else:
            return {"available": False, "error": "Invalid date/time format"}
    
    end_dt = dt + timedelta(minutes=duration_minutes)
    date_key = dt.date().isoformat()
    
    # Check conflicts
    if date_key in MOCK_CALENDAR:
        for event in MOCK_CALENDAR[date_key]:
            event_start = datetime.fromisoformat(event['start'])
            event_end = datetime.fromisoformat(event['end'])
            if not (end_dt <= event_start or dt >= event_end):
                return {"available": False, "conflict": event}
    
    return {"available": True, "slot": dt.isoformat(), "end": end_dt.isoformat()}

def get_alternative_slots(preferred_dt: datetime, num_alternatives: int = 3) -> List[Dict]:
    """Get alternative time slots"""
    alternatives = []
    date_key = preferred_dt.date().isoformat()
    
    # Check same day - morning, afternoon, evening
    base_times = [
        preferred_dt.replace(hour=9, minute=0),
        preferred_dt.replace(hour=11, minute=0),
        preferred_dt.replace(hour=14, minute=0),
        preferred_dt.replace(hour=16, minute=0),
    ]
    
    for alt_dt in base_times:
        if alt_dt != preferred_dt:
            result = check_calendar_availability(
                alt_dt.date().isoformat(), 
                alt_dt.time().isoformat()[:5]
            )
            if result.get('available'):
                alternatives.append({
                    "date": alt_dt.date().isoformat(),
                    "time": alt_dt.time().isoformat()[:5],
                    "datetime": alt_dt.isoformat()
                })
                if len(alternatives) >= num_alternatives:
                    break
                    
    # Check next few days
    if len(alternatives) < num_alternatives:
        for days_ahead in range(1, 4):
            next_date = preferred_dt + timedelta(days=days_ahead)
            for hour in [10, 11, 14, 15, 16]:
                alt_dt = next_date.replace(hour=hour, minute=0)
                result = check_calendar_availability(
                    alt_dt.date().isoformat(),
                    alt_dt.time().isoformat()[:5]
                )
                if result.get('available'):
                    alternatives.append({
                        "date": alt_dt.date().isoformat(),
                        "time": alt_dt.time().isoformat()[:5],
                        "datetime": alt_dt.isoformat()
                    })
                    if len(alternatives) >= num_alternatives:
                        break
            if len(alternatives) >= num_alternatives:
                break
                
    return alternatives[:num_alternatives]

def book_calendar_event(event_data: Dict) -> Dict:
    """Book a calendar event using Google Calendar API"""
    service = _get_calendar_service()
    
    if service:
        try:
            event_body = {
                'summary': event_data.get('summary', 'Property Visit'),
                'description': event_data.get('description', ''),
                'start': {'dateTime': event_data['start'], 'timeZone': 'Asia/Karachi'},
                'end': {'dateTime': event_data['end'], 'timeZone': 'Asia/Karachi'},
                'attendees': event_data.get('attendees', []),
                'reminders': {'useDefault': False, 'overrides': [
                    {'method': 'email', 'minutes': 1440},
                    {'method': 'popup', 'minutes': 30}
                ]}
            }
            event = service.events().insert(calendarId='primary', body=event_body, sendUpdates='all').execute()
            return {"success": True, "event_id": event['id'], "event": event}
        except Exception as e:
            logger.error(f"Google Calendar API error: {e}")
            # Fall back to mock
    
    # Mock fallback
    dt = datetime.fromisoformat(event_data['start'])
    date_key = dt.date().isoformat()
    if date_key not in MOCK_CALENDAR:
        MOCK_CALENDAR[date_key] = []
    event_id = f"evt_{len(MOCK_CALENDAR[date_key]) + 1}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    event = {
        'id': event_id,
        'start': event_data['start'],
        'end': event_data['end'],
        'summary': event_data.get('summary', 'Property Visit'),
        'description': event_data.get('description', ''),
        'attendees': event_data.get('attendees', [])
    }
    MOCK_CALENDAR[date_key].append(event)
    return {"success": True, "event_id": event_id, "event": event}

def cancel_calendar_event(event_id: str) -> Dict:
    """Cancel a calendar event"""
    for date_key, events in MOCK_CALENDAR.items():
        for i, event in enumerate(events):
            if event['id'] == event_id:
                MOCK_CALENDAR[date_key].pop(i)
                return {"success": True, "message": "Event cancelled"}
    return {"success": False, "error": "Event not found"}

def reschedule_calendar_event(event_id: str, new_start: str, new_end: str) -> Dict:
    """Reschedule a calendar event"""
    # Cancel old
    cancel_result = cancel_calendar_event(event_id)
    if not cancel_result['success']:
        return cancel_result
        
    # Book new
    return book_calendar_event({
        'start': new_start,
        'end': new_end,
        'summary': 'Property Visit (Rescheduled)',
        'description': 'Rescheduled appointment'
    })

# ============================================================================
# EMAIL TOOLS
# ============================================================================

def send_email(to_email: str, subject: str, body: str, from_email: str = None) -> Dict:
    """Send an email over SMTP when configured, otherwise log a demo message.

    Set ``SMTP_HOST``, ``SMTP_USER`` and ``SMTP_PASSWORD`` (plus optionally
    ``SMTP_PORT`` and ``AGENT_EMAIL``) to enable real delivery. Without them the
    function is a safe no-op, which keeps local development and CI fully
    offline.
    """
    logger.info("EMAIL TO: %s", to_email)
    logger.info("SUBJECT: %s", subject)
    logger.info("BODY: %s...", body[:200])

    sender_email = from_email or SMTP_USER or AGENT_EMAIL
    if not (SMTP_HOST and sender_email and SMTP_PASSWORD):
        return {
            "success": True,
            "message": "Email logged (SMTP not configured)",
            "demo": True,
        }

    try:
        msg = MIMEMultipart()
        msg["From"] = sender_email
        msg["To"] = to_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain", "utf-8"))

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10) as server:
            server.starttls()
            server.login(sender_email, SMTP_PASSWORD)
            server.send_message(msg)
        return {"success": True, "message": "Email sent"}
    except Exception as exc:  # noqa: BLE001 - email must never break a booking
        logger.warning("Email delivery failed: %s", exc)
        return {"success": False, "message": f"Email failed: {exc}"}

def send_agent_notification(agent_email: str, event_data: Dict) -> Dict:
    """Send appointment notification to agent"""
    subject = f"New Property Visit: {event_data.get('property_title', 'Property')}"
    
    body = f"""
Assalam-o-Alaikum,

Aap ko nayi property visit schedule ki gayi hai:

**Client Details:**
- Name: {event_data.get('client_name', 'N/A')}
- Phone: {event_data.get('client_phone', 'N/A')}
- Email: {event_data.get('client_email', 'N/A')}
- Requirements: {event_data.get('client_requirements', 'N/A')}

**Property:**
- Title: {event_data.get('property_title', 'N/A')}
- Address: {event_data.get('property_address', 'N/A')}
- Price: PKR {event_data.get('property_price', 0):,}

**Appointment:**
- Date: {event_data.get('appointment_date', 'N/A')}
- Time: {event_data.get('appointment_time', 'N/A')}
- Duration: 60 minutes

**Meeting Notes:**
{event_data.get('meeting_notes', 'N/A')}

Please prepare property documents and contact client 30 minutes before visit.

Regards,
RealEstate Hub Automation
"""
    return send_email(agent_email, subject, body)

def send_client_confirmation(client_email: str, client_name: str, event_data: Dict) -> Dict:
    """Send confirmation to client"""
    subject = f"Property Visit Confirmed - {event_data.get('property_title', 'Property')}"
    
    body = f"""
Assalam-o-Alaikum {client_name},

Aap ki property visit confirm ho gayi hai:

**Property:** {event_data.get('property_title', 'N/A')}
**Address:** {event_data.get('property_address', 'N/A')}
**Date:** {event_data.get('appointment_date', 'N/A')}
**Time:** {event_data.get('appointment_time', 'N/A')}

Hamara agent aap se 30 minutes pehle contact karega.
Koi sawal ho to humein call karein.

Shukriya,
RealEstate Hub Team
"""
    return send_email(client_email, subject, body)

# ============================================================================
# PROPERTY SEARCH TOOL
# ============================================================================

from real_estate_agent.recommendation import PropertyRecommendationEngine, UserPreferences, parse_preferences_from_text

# Global engine instance
_recommendation_engine = None

def get_recommendation_engine():
    global _recommendation_engine
    if _recommendation_engine is None:
        _recommendation_engine = PropertyRecommendationEngine()
    return _recommendation_engine

def search_properties(query: str, budget_max: int = None, city: str = None, 
                      area: str = None, min_bedrooms: int = None,
                      property_type: str = None, purpose: str = None,
                      top_k: int = 5) -> List[Dict]:
    """Search properties using recommendation engine"""
    engine = get_recommendation_engine()
    
    prefs = UserPreferences(
        budget_max=budget_max,
        city=city,
        area=area,
        min_bedrooms=min_bedrooms,
        property_type=property_type,
        purpose=purpose
    )
    
    results = engine.recommend(prefs, top_k=top_k)
    return results

def search_properties_from_text(text: str, top_k: int = 5) -> List[Dict]:
    """Search properties by parsing natural language text"""
    prefs_dict = parse_preferences_from_text(text)
    
    prefs = UserPreferences(
        budget_min=prefs_dict.get('budget_min'),
        budget_max=prefs_dict.get('budget_max'),
        city=prefs_dict.get('city'),
        area=prefs_dict.get('area'),
        min_bedrooms=prefs_dict.get('min_bedrooms', 0),
        property_type=prefs_dict.get('property_type'),
        purpose=prefs_dict.get('purpose'),
        amenities=prefs_dict.get('amenities', []),
        investment_goals=prefs_dict.get('investment_goals')
    )
    
    engine = get_recommendation_engine()
    return engine.recommend(prefs, top_k=top_k)


# ============================================================================
# RAG SEARCH TOOL
# ============================================================================

_rag_pipeline = None

def get_rag_pipeline():
    global _rag_pipeline
    if _rag_pipeline is None:
        from real_estate_agent.rag_pipeline import RealEstateRAG
        _rag_pipeline = RealEstateRAG()
        _rag_pipeline.load_data()
        _rag_pipeline.initialize_vector_store()
    return _rag_pipeline

def rag_search(query: str, top_k: int = 5) -> List[Dict]:
    """Search knowledge base for factual questions"""
    rag = get_rag_pipeline()
    return rag.search(query, top_k=top_k)


# ============================================================================
# APPOINTMENT MANAGEMENT
# ============================================================================

def book_appointment(client_name: str, client_phone: str, client_email: str,
                     property_id: str, date_str: str, time_str: str,
                     duration_minutes: int = 60, notes: str = "",
                     agent_email: str = AGENT_EMAIL or "agent@realestatehub.com") -> Dict:
    """Book a property visit appointment"""
    # Check availability
    avail = check_calendar_availability(date_str, time_str, duration_minutes)
    if not avail.get('available'):
        alternatives = get_alternative_slots(
            datetime.fromisoformat(avail.get('slot', datetime.now().isoformat()))
        )
        return {
            "success": False,
            "error": "Slot not available",
            "alternatives": alternatives
        }
    
    # Get property details
    from real_estate_agent.rag_pipeline import RealEstateRAG
    rag = RealEstateRAG()
    rag.load_data()
    prop = rag.get_property_details(property_id)
    
    if not prop:
        return {"success": False, "error": "Property not found"}
    
    # Book calendar event
    start_dt = datetime.fromisoformat(avail['slot'])
    end_dt = start_dt + timedelta(minutes=duration_minutes)
    
    event_data = {
        'start': start_dt.isoformat(),
        'end': end_dt.isoformat(),
        'summary': f"Property Visit: {client_name} - {prop['title']}",
        'description': f"Client: {client_name}\nPhone: {client_phone}\nProperty: {prop['title']}\nNotes: {notes}",
        'attendees': [
            {'email': agent_email, 'displayName': 'Agent'},
            {'email': client_email, 'displayName': client_name}
        ]
    }
    
    cal_result = book_calendar_event(event_data)
    if not cal_result['success']:
        return {"success": False, "error": "Failed to book calendar"}
    
    # Send notifications
    event_data_for_email = {
        'client_name': client_name,
        'client_phone': client_phone,
        'client_email': client_email,
        'client_requirements': notes,
        'property_title': prop['title'],
        'property_address': f"{prop['area']}, {prop['city']}",
        'property_price': prop['price'],
        'appointment_date': start_dt.strftime("%Y-%m-%d"),
        'appointment_time': start_dt.strftime("%H:%M"),
        'meeting_notes': notes,
        'calendar_link': f"https://calendar.google.com/event?eid={cal_result['event_id']}"
    }
    
    send_agent_notification(AGENT_EMAIL or "agent@realestatehub.com", event_data_for_email)
    send_client_confirmation(client_email, client_name, event_data_for_email)
    
    return {
        "success": True,
        "event_id": cal_result['event_id'],
        "appointment": {
            "date": start_dt.strftime("%Y-%m-%d"),
            "time": start_dt.strftime("%H:%M"),
            "property": prop['title'],
            "address": f"{prop['area']}, {prop['city']}"
        }
    }

def reschedule_appointment(event_id: str, new_date: str, new_time: str) -> Dict:
    """Reschedule an appointment"""
    new_start = f"{new_date}T{new_time}"
    new_end_dt = datetime.fromisoformat(new_start) + timedelta(hours=1)
    new_end = new_end_dt.isoformat()
    
    result = reschedule_calendar_event(event_id, new_start, new_end)
    if result.get('success'):
        # Send notifications
        return {"success": True, "message": "Appointment rescheduled", "new_time": new_start}
    return result

def cancel_appointment(event_id: str, reason: str = "Client requested cancellation") -> Dict:
    """Cancel an appointment"""
    result = cancel_calendar_event(event_id)
    if result.get('success'):
        # Notify agent and client
        return {"success": True, "message": "Appointment cancelled", "reason": reason}
    return result


if __name__ == "__main__":
    # Test tools
    print("Testing tools...")
    
    # Test property search
    results = search_properties_from_text("3 crore budget, Lahore DHA, 3 bedroom")
    print(f"Found {len(results)} properties")
    for r in results[:3]:
        print(f"  {r['title']} - {r['price_formatted']}")
    
    # Test calendar
    avail = check_calendar_availability("2024-01-20", "10:00")
    print(f"Calendar available: {avail}")
    
    # Test booking
    book_result = book_appointment(
        client_name="Ahmed Khan",
        client_phone="0300-1234567",
        client_email="ahmed@example.com",
        property_id="prop_001",
        date_str="2024-01-20",
        time_str="10:00"
    )
    print(f"Booking result: {book_result}")