import requests
import streamlit as st

st.set_page_config(page_title="Personal Agent", page_icon=":robot_face:")
st.title("Personal Agent")
st.write("A Personal Agent built using LangChain, LangGraph, and FastAPI")

# -------------------------
# Configuration
# -------------------------
API_URL = "http://127.0.0.1:8000/chat"

# These should stay in sync with backend.py / ai_agent.py
GOOGLE_MODELS = ["gemini-1.5-flash"]
GROQ_MODELS = ["llama-3.3-70b-versatile"]

# -------------------------
# Sidebar Controls
# -------------------------
st.sidebar.header("Model & Tools")
provider = st.sidebar.radio("Select Provider", options=["google", "groq"], index=0)

if provider == "google":
    model_name = st.sidebar.selectbox("Select Google Model", GOOGLE_MODELS)
else:
    model_name = st.sidebar.selectbox("Select Groq Model", GROQ_MODELS)

allow_search = st.sidebar.checkbox("Enable Web Search Tool", value=True)

# -------------------------
# Main Layout
# -------------------------
st.subheader("System Prompt")
system_prompt = st.text_area(
    "Define your AI Agent",
    height=100,
    placeholder="Act as a smart, friendly, and helpful AI assistant...",
)

st.subheader("Chat")
user_query = st.text_area(
    "Your Message",
    height=150,
    placeholder="Ask your personal agent anything...",
)

# -------------------------
# Session State for History
# -------------------------
if "history" not in st.session_state:
    st.session_state.history = []

col1, col2 = st.columns([1, 3])

with col1:
    send_clicked = st.button("Send")
with col2:
    clear_clicked = st.button("Clear Conversation")

if clear_clicked:
    st.session_state.history = []

# -------------------------
# Call Backend
# -------------------------
if send_clicked:
    if not user_query.strip():
        st.warning("Please enter a message before sending.")
    else:
        payload = {
            "model_name": model_name,
            "model_provider": provider,
            "system_prompt": system_prompt or "You are a helpful personal AI agent.",
            "messages": user_query,
            "allow_search": allow_search,
        }

        try:
            resp = requests.post(API_URL, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()

            if "error" in data:
                st.error(f"Backend error: {data['error']}")
            else:
                ai_answer = data.get("response", "")
                st.session_state.history.append(("User", user_query))
                st.session_state.history.append(("Agent", ai_answer))
        except Exception as e:
            st.error(f"Request failed: {e}")

# -------------------------
# Display Conversation
# -------------------------
if st.session_state.history:
    st.subheader("Conversation")
    for role, text in st.session_state.history:
        if role == "User":
            st.markdown(f"**🧑 You:** {text}")
        else:
            st.markdown(f"**🤖 Agent:** {text}")
