import streamlit as st
from embed.embed_chunks import MilvusManager
from llm.groq_client import GroqClient
from llm.gemini_client import GoogleGenAIClient

@st.cache_resource
def initialize_system(model_choice):
    milvus = MilvusManager()
    if model_choice == "Groq":
        return GroqClient(milvus)
    elif model_choice == "Gemini":
        return GoogleGenAIClient(milvus)
    else:
        raise ValueError("Invalid model choice")

# Sidebar
with st.sidebar:
    st.header("Configuration")
    model_choice = st.selectbox(
        "Choose LLM",
        ("Groq", "Gemini"),
        index=0,
        help="Select which language model to use for responses"
    )

# Initialize the selected client
client = initialize_system(model_choice)

# UI
st.title('Ellie.ai Doc Assistant')
st.markdown(f"Powered by {model_choice} & Milvus")

# Clear chat history when switching models
if 'current_model' not in st.session_state:
    st.session_state.current_model = model_choice
elif st.session_state.current_model != model_choice:
    st.session_state.messages = []
    st.session_state.current_model = model_choice

if 'messages' not in st.session_state:
    st.session_state.messages = []

if prompt := st.chat_input("Ask about Ellie.ai's documentation"):
    st.session_state.messages.append({'role':'user', "content": prompt})

    with st.spinner('Thinking...'):
        response = client.generate_response(prompt)

    st.session_state.messages.append({'role':'assistant', 'content': response})

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])