import streamlit as st
from embed.embed_chunks import MilvusManager
from llm.groq_client import GroqClient

@st.cache_resource
def initialize_system():
    milvus = MilvusManager()
    return GroqClient(milvus)

groq_client = initialize_system()

# UI
st.title('Ellie.ai Doc Assistant')
st.markdown("Powered by Groq LPU & Milvus")

if 'messages' not in st.session_state:
    st.session_state.messages=[]

if prompt := st.chat_input("Ask about Ellie.ai's documentation"):
    st.session_state.messages.append({'role':'user', "content": prompt})

    with st.spinner('Thinking...'):
        response = groq_client.generate_response(prompt)

    st.session_state.messages.append({'role':'assistant', 'content': response})


for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])