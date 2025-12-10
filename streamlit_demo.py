#!/usr/bin/env python3
"""
Streamlit Demo for Drug Repurposing Chat System
"""

import streamlit as st
import requests
import json
import time
from typing import List, Dict, Any

# FastAPI server configuration
API_BASE_URL = "http://localhost:8000"

# Available drugs
AVAILABLE_DRUGS = {
    "aspirin": "Aspirin - Cancer Prevention & Cardiovascular",
    "apomorphine": "Apomorphine - Parkinson's & Addiction Treatment",
    "insulin": "Insulin - Metabolic & Research Applications"
}

def init_session_state():
    """Initialize session state for chat history"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_drug" not in st.session_state:
        st.session_state.current_drug = "aspirin"
    if "session_id" not in st.session_state:
        st.session_state.session_id = f"streamlit_{int(time.time())}"

def make_chat_request(drug_id: str, message: str, session_id: str, conversation_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
    """Make a chat request to the FastAPI backend"""
    payload = {
        "session_id": session_id,
        "drug_id": drug_id,
        "message": message
    }

    # Add conversation history if available
    if conversation_history:
        payload["conversation_history"] = conversation_history

    try:
        response = requests.post(f"{API_BASE_URL}/chat", json=payload, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to connect to chat service: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}

def display_source(source: Dict[str, Any], index: int) -> None:
    """Display a source with proper formatting"""
    # Generate truly unique key using timestamp and random component
    import time
    import random
    unique_key = f"preview_{source['doc_id']}_{index}_{int(time.time()*1000)}_{random.randint(1000,9999)}"

    with st.expander(f"📄 {source['doc_title']} (Distance: {source['distance']:.3f})", expanded=False):
        st.write(f"**Document ID:** {source['doc_id']}")
        st.write(f"**Text Preview:**")
        st.text_area("", source['text_preview'], height=100, disabled=True, key=unique_key)

def display_chat_message(message: Dict[str, Any], is_user: bool = False):
    """Display a chat message with proper formatting"""
    if is_user:
        with st.chat_message("user"):
            st.write(message["content"])
    else:
        with st.chat_message("assistant"):
            st.write(message["answer"])

            if "sources" in message and message["sources"]:
                st.write("**📚 Sources:**")
                for i, source in enumerate(message["sources"]):
                    display_source(source, i)

def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="Drug Repurposing Chat",
        page_icon="💊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("💊 Drug Repurposing Research Chat")
    st.markdown("*Powered by RAG (Retrieval-Augmented Generation) from Scientific Literature*")
    st.markdown("*Explore repurposing opportunities for aspirin, apomorphine, and insulin*")

    # Initialize session state
    init_session_state()

    # Sidebar for drug selection and info
    with st.sidebar:
        st.header("🔬 Drug Selection")

        # Drug selector for pre-loaded drugs
        selected_drug_display = st.selectbox(
            "Choose a drug to discuss:",
            options=list(AVAILABLE_DRUGS.keys()),
            format_func=lambda x: AVAILABLE_DRUGS[x],
            index=list(AVAILABLE_DRUGS.keys()).index(st.session_state.current_drug),
            key="drug_selector"
        )

        # Update current drug if changed
        if selected_drug_display != st.session_state.current_drug:
            st.session_state.current_drug = selected_drug_display
            st.session_state.messages = []  # Clear chat when switching drugs
            st.session_state.session_id = f"streamlit_{int(time.time())}"

            if st.session_state.current_drug and st.session_state.current_drug not in AVAILABLE_DRUGS:
                st.info(f"📊 Currently analyzing: **{st.session_state.current_drug.title()}**")
                st.warning("⚠️ This drug hasn't been processed yet. In full implementation, this would trigger automatic PDF download and RAG ingestion.")

        st.markdown("---")

        # Current drug info
        if st.session_state.current_drug in AVAILABLE_DRUGS:
            drug_name = AVAILABLE_DRUGS[st.session_state.current_drug].split(' - ')[0]
            drug_focus = AVAILABLE_DRUGS[st.session_state.current_drug].split(' - ')[1]
        else:
            # Custom drug
            drug_name = st.session_state.current_drug.title()
            drug_focus = "Custom Drug Analysis (Processing Required)"

        st.subheader(f"Current Drug: {drug_name}")
        st.write(f"**Focus:** {drug_focus}")

        # System status
        st.markdown("---")
        st.subheader("🔍 System Status")

        try:
            health_response = requests.get(f"{API_BASE_URL}/health", timeout=5)
            if health_response.status_code == 200:
                st.success("✅ Chat service is online")
            else:
                st.error("❌ Chat service is offline")
        except:
            st.error("❌ Cannot connect to chat service")
            st.info("💡 **Required**: Run `python -m app.main` in a separate terminal")
            st.info("🔗 This starts the FastAPI backend for chat functionality")

        # Available drugs list
        st.markdown("---")
        st.subheader("📚 Available Drugs")

        for drug_id, description in AVAILABLE_DRUGS.items():
            drug_name = description.split(' - ')[0]
            drug_focus = description.split(' - ')[1]
            st.write(f"• **{drug_name}**: {drug_focus}")

    # Main chat interface
    drug_display = AVAILABLE_DRUGS[st.session_state.current_drug].split(' - ')[0]
    st.header(f"💬 Chat about {drug_display} Repurposing")

    # Display chat history
    for message in st.session_state.messages:
        display_chat_message(message, message.get("is_user", False))

    # Chat input
    placeholder_text = f"Ask about {drug_display} repurposing..."
    if prompt := st.chat_input(placeholder_text):
        # Add user message to history
        user_message = {"content": prompt, "is_user": True}
        st.session_state.messages.append(user_message)
        display_chat_message(user_message, True)

        # Build conversation history for API (exclude current message)
        conversation_history = []
        for msg in st.session_state.messages[:-1]:  # Exclude the current user message
            if msg.get("is_user"):
                conversation_history.append({"role": "user", "content": msg["content"]})
            elif "answer" in msg:
                conversation_history.append({"role": "assistant", "content": msg["answer"]})

        # Make API request with conversation history
        with st.spinner("🔍 Searching research documents..."):
            response = make_chat_request(
                st.session_state.current_drug,
                prompt,
                st.session_state.session_id,
                conversation_history
            )

        # Handle response
        if "error" in response:
            error_message = {
                "answer": f"❌ Error: {response['error']}",
                "sources": []
            }
            st.session_state.messages.append(error_message)
            display_chat_message(error_message, False)
        else:
            # Add assistant response to history
            st.session_state.messages.append(response)
            display_chat_message(response, False)

    # Footer
    st.markdown("---")
    st.markdown("*Built with FastAPI, ChromaDB, Google Gemini, and Streamlit*")
    st.markdown("*RAG system retrieves and synthesizes information from scientific research papers*")

if __name__ == "__main__":
    main()
