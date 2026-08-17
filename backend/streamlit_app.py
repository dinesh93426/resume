import requests
import streamlit as st

st.set_page_config(page_title="AI Resume Assistant", page_icon="📄")
st.title("AI Resume Assistant")

BACKEND_URL = "http://127.0.0.1:8000"

uploaded_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])
if st.button("Process Resume"):
    if uploaded_file is None:
        st.warning("Please upload a PDF first.")
    else:
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
        response = requests.post(f"{BACKEND_URL}/upload", files=files, timeout=120)
        if response.ok:
            st.success("Resume processed and stored.")
            st.json(response.json())
        else:
            st.error(response.text)

question = st.text_input("Ask a question about the resume")
if st.button("Ask"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        response = requests.post(
            f"{BACKEND_URL}/search",
            json={"question": question},
            timeout=120,
        )
        if response.ok:
            data = response.json()
            st.subheader("Answer")
            st.write(data.get("answer", ""))
            st.subheader("Retrieved Context")
            st.write(data.get("relevant_chunks", []))
        else:
            st.error(response.text)
