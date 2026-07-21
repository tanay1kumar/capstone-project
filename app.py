import streamlit as st
from dotenv import load_dotenv

from rag.query import RAGQuery

load_dotenv()

st.set_page_config(
    page_title="SEC Regulations Assistant",
    page_icon="⚖️",
    layout="centered",
)

st.title("SEC Regulations Assistant")
st.caption("Ask questions about SEC rules and regulations. Answers are grounded in official documents only.")

# cache so the model doesn't reload on every message
@st.cache_resource
def load_rag():
    return RAGQuery()

rag = load_rag()

# session_state keeps chat history across streamlit reruns
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- **{s['file']}**, page {s['page']}")

if question := st.chat_input("Ask about SEC regulations..."):
    with st.chat_message("user"):
        st.markdown(question)

    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        with st.spinner("Searching regulations..."):
            result = rag.ask(question)

        st.markdown(result["answer"])

        if result["sources"]:
            with st.expander("Sources"):
                for s in result["sources"]:
                    st.markdown(f"- **{s['file']}**, page {s['page']}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
    })
