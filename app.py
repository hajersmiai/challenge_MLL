import streamlit as st
import requests
import json
from dotenv import load_dotenv
import os
import asyncio
import time

# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="Detective Translator", layout="centered")

# Load environment variables from .env file
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

# -----------------------------
# LLM call (async)
# -----------------------------
async def translate_text(text: str, target_language: str) -> str:
    """
    Translates text from detective slang to standard English, then to the target language.
    """
    if not API_KEY:
        st.error("API Key not found. Please create a .env file with GEMINI_API_KEY='YOUR_API_KEY_HERE'")
        return "Error: API Key not configured."

    prompt = (
        "First, translate the following text from 'detective slang' into standard, formal language. "
        f"Then, translate the result into {target_language}. "
        f"The original text is: '{text}'"
    )

    chat_history = [{"role": "user", "parts": [{"text": prompt}]}]
    payload = {"contents": chat_history}
    api_url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.5-flash-preview-05-20:generateContent?key={API_KEY}"
    )

    # Exponential backoff
    max_retries = 5
    delay = 1
    for attempt in range(max_retries):
        try:
            response = requests.post(
                api_url,
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload),
                timeout=30,
            )
            # If status is bad, raise for_status will trigger except below
            response.raise_for_status()

            data = response.json()
            # Expected structure: candidates[0].content.parts[0].text
            candidates = data.get("candidates") or []
            if candidates and "content" in candidates[0]:
                parts = candidates[0]["content"].get("parts") or []
                if parts and "text" in parts[0]:
                    return parts[0]["text"].strip()

            return "Translation failed: Unexpected API response structure."

        except requests.exceptions.HTTPError as http_err:
            if attempt < max_retries - 1:
                st.warning(
                    f"Request failed with status code {response.status_code if 'response' in locals() else '??'}. "
                    f"Retrying in {delay} seconds..."
                )
                time.sleep(delay)
                delay *= 2
            else:
                st.error(f"HTTP error: {http_err}")
                return "Translation failed due to HTTP error."
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                st.warning(f"Network error. Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2
            else:
                st.error(f"Network error: {e}")
                return "An error occurred. Please check your internet connection or API key."

    return "Translation failed after multiple retries."

# -----------------------------
# UI
# -----------------------------
st.markdown(
    "<h1 style='text-align: center; color: #1f2937; font-weight: bold;'>Detective Language Translator</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align: center; color: #4b5563;'>Translate classic detective slang into formal language, and then into a language of your choice.</p>",
    unsafe_allow_html=True,
)

st.subheader("Enter your detective phrase:")
detective_input = st.text_area(
    label="",
    placeholder="e.g., The perp fled the scene with the loot.",
    height=150
)

col1, col2 = st.columns([3, 1])
with col1:
    st.subheader("Choose a target language:")
    language_select = st.selectbox(
        label="",
        options=("English", "Spanish", "French", "German", "Japanese", "Chinese (Simplified)"),
        index=0
    )

# Prepare the state for the result
if "translated_text" not in st.session_state:
    st.session_state.translated_text = ""

with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Translate", use_container_width=True, type="primary"):
        if not detective_input.strip():
            st.warning("Please enter some text to translate.")
        else:
            with st.spinner("Translating..."):
                # Execute properly the async in Streamlit
                result = asyncio.run(translate_text(detective_input.strip(), language_select))
                st.session_state.translated_text = result

# Output
st.subheader("Translated text:")
st.markdown(
    f"<div style='background-color: #f9fafb; padding: 16px; border-radius: 8px; border: 1px solid #d1d5db; min-height: 100px;'>{st.session_state.translated_text or ''}</div>",
    unsafe_allow_html=True
)

st.markdown("---")
st.subheader("Need ideas? Try these phrases:")
st.markdown("""
- "The gumshoe was on the trail of a known stool pigeon."
- "It was a classic whodunit with more red herrings than a fish market."
- "He got pinched for roughing up a dame and casing a joint."
- "Let's go knock on the door of the head honcho."
""")