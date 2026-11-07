"""
Streamlit UI for Real Estate Voice Agent
Run with: streamlit run ui.py
"""
import streamlit as st
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from real_estate_agent.agent import run_agent
from real_estate_agent.tools import rag_search, search_properties_from_text

st.set_page_config(page_title="RealEstate Hub - AI Voice Agent", page_icon="🏠", layout="wide")

st.title("🏠 RealEstate Hub - AI Voice Agent")
st.markdown("### Production-Ready Real Estate Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None

with st.sidebar:
    st.header("About")
    st.info("AI voice agent for real estate. Ask about properties, prices, payment plans, or book a visit.")
    st.header("Quick Actions")
    if st.button("Buy Property"):
        st.session_state.messages.append({"role": "user", "content": "Main ghar dhundh raha hoon, budget 3 crore"})
        st.rerun()
    if st.button("Rent Property"):
        st.session_state.messages.append({"role": "user", "content": "Main rental property dhundh raha hai"})
        st.rerun()
    if st.button("Investment"):
        st.session_state.messages.append({"role": "user", "content": "Main investment karna chahta hoon"})
        st.rerun()
    if st.button("Book Visit"):
        st.session_state.messages.append({"role": "user", "content": "Main property visit book karna chahta hoon"})
        st.rerun()
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.session_state.conversation_id = None
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Apna sawal yahan likhein..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Soch raha hoon..."):
            history = [m for m in st.session_state.messages[:-1]]
            result = run_agent(prompt, st.session_state.conversation_id, history)
            st.session_state.conversation_id = result.get("conversation_id")
            response = result.get("response", "Maazrat, samajh nahi aaya. Dobara bhejein.")
            st.markdown(response)
            if result.get("properties"):
                st.subheader("Recommended Properties")
                for p in result["properties"]:
                    with st.expander(f"{p['title']} - {p['price_formatted']}"):
                        st.write(f"Type: {p['type']}")
                        st.write(f"Area: {p['area_sqft']} sqft")
                        if p.get('monthly_installment'):
                            st.write(f"Monthly: PKR {p['monthly_installment']:,.0f}")
                        st.write(f"Score: {p['score']:.1f}")
            if result.get("needs_clarification"):
                st.warning(result.get("clarification_question", "Kuch aur detail chahiye."))
    st.session_state.messages.append({"role": "assistant", "content": response})

st.markdown("---")
st.caption("RealEstate Hub AI Voice Agent | Powered by LangGraph + RAG")