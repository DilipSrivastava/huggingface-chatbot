import os

import streamlit as st
from dotenv import load_dotenv
from huggingface_hub import InferenceClient


load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

client = InferenceClient(
    api_key=HF_TOKEN
)

MODEL_NAME = "Qwen/Qwen3.6-27B"


st.set_page_config(
    page_title="Hugging Face Chatbot",
    page_icon="🤗"
)

st.title("🤗 Hugging Face Chatbot")
st.caption(f"We are using: {MODEL_NAME}")


# -----------------------------
# Initialize chat history
# -----------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# -----------------------------
# Show previous messages
# -----------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# -----------------------------
# User input
# -----------------------------

prompt = st.chat_input(
    "Ask me anything..."
)


if prompt:

    # Show user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Save user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )


    # Call Hugging Face model
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=st.session_state.messages,
        max_tokens=500,
        temperature=0.7
    )


    assistant_reply = response.choices[0].message.content


    # Show assistant response
    with st.chat_message("assistant"):
        st.markdown(assistant_reply)


    # Save assistant response
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": assistant_reply
        }
    )