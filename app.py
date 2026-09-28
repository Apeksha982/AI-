import streamlit as st
import os
from agent import run_agent
from tools import build_index_from_pdf


st.set_page_config(page_title="Agentic PDF Assistant", page_icon="")
st.title("Your Assistant")
st.caption("ask questions.")

uploaded = st.file_uploader("Upload a PDF", type="pdf")
if uploaded is not None and st.session_state.get("last_file") != uploaded.name:
    with st.spinner("Indexing PDF..."):
        n = build_index_from_pdf(uploaded)
    st.session_state["last_file"] = uploaded.name
    st.success(f"Indexed {n} chunks from {uploaded.name}")

if "history" not in st.session_state:
    st.session_state.history = []

question = st.chat_input("Ask something...")

for entry in st.session_state.history:
    with st.chat_message(entry["role"]):
        st.write(entry["content"])
        if entry.get("steps"):
            with st.expander("reasoning"):
                for step in entry["steps"]:
                    st.write("- " + step)

if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, steps = run_agent(question)
        st.write(answer)
        if steps:
            with st.expander("Agent reasoning"):
                for step in steps:
                    st.write("- " + step)
        else:
            st.caption("No tools were needed for this answer.")
    st.session_state.history.append({"role": "assistant", "content": answer, "steps": steps})
