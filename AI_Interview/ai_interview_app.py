import streamlit as st
from mistralai import Mistral
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import faiss
import numpy as np
import tempfile
import os

# ---------------- UI ----------------
st.set_page_config(page_title="AI Interview App", layout="centered")
st.title("📄 Resume–JD Match & Interview Generator")

st.markdown(
    "Upload Resume & Job Description PDFs. "
    "Get ATS match % and interview questions."
)

# ---------------- Inputs ----------------
api_key = st.text_input(
    "Enter your Mistral API Key",
    type="password",
    help="Your API key is never stored"
)

resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
jd_file = st.file_uploader("Upload Job Description (PDF)", type=["pdf"])

MODEL = "mistral-small-latest"

# ---------------- Utility Functions ----------------
def save_file(uploaded_file):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp.write(uploaded_file.read())
    tmp.close()
    return tmp.name

def load_pdf_text(path):
    reader = PdfReader(path)
    return " ".join(
        page.extract_text() for page in reader.pages if page.extract_text()
    )

def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap
    return chunks

# ---------------- Main Logic ----------------
if st.button("Analyze"):
    if not api_key or not resume_file or not jd_file:
        st.error("Please provide API key, Resume, and JD.")
        st.stop()

    with st.spinner("Analyzing documents..."):
        # ---- Save PDFs ----
        resume_path = save_file(resume_file)
        jd_path = save_file(jd_file)

        resume_text = load_pdf_text(resume_path)
        jd_text = load_pdf_text(jd_path)

        os.remove(resume_path)
        os.remove(jd_path)

        # ---- Chunk & Embed ----
        chunks = chunk_text(resume_text + "\n" + jd_text)

        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = embedder.encode(chunks)

        # ---- FAISS Index ----
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(np.array(embeddings))

        def retrieve_context(query, k=6):
            q_emb = embedder.encode([query])
            _, idx = index.search(q_emb, k)
            return "\n".join(chunks[i] for i in idx[0])

        client = Mistral(api_key=api_key)

        # ---------------- MATCH % ----------------
        match_context = retrieve_context(
            "resume skills experience projects job description requirements"
        )

        match_prompt = f"""
You are an ATS system.

Calculate the percentage match (0 to 100) between the resume and job description.
Consider:
- Skills overlap
- Tools & technologies
- Experience relevance
- Projects alignment

Return ONLY a number.

CONTEXT:
{match_context}
"""

        resp = client.chat.complete(
            model=MODEL,
            messages=[{"role": "user", "content": match_prompt}]
        )

        match_pct = float(resp.choices[0].message.content.strip())
        st.success(f"✅ Resume–JD Match: {match_pct}%")

        # ---------------- QUESTIONS ----------------
        if match_pct < 60:
            st.warning("❌ Match below 60%. Candidate not shortlisted.")
        else:
            st.subheader("📌 Interview Questions")

            q_context = retrieve_context(
                "technical skills projects experience scenarios"
            )

            question_prompt = f"""
                    You are a technical interviewer.

                    Using the CONTEXT:
                    - Job description requirements
                    - Skills mentioned in resume
                    - Projects done by candidate

                    Generate:
                    1. 5 technical questions on the job description
                    2. 3 project-based questions on the projects done by candidate
                    3. 2 skill-based questions on the skills mentioned in resume
                    3. 2 scenario-based questions based on the job description
                    CONTEXT:
                    {q_context}
                    """

            resp = client.chat.complete(
                model=MODEL,
                messages=[{"role": "user", "content": question_prompt}]
            )

            st.markdown(resp.choices[0].message.content)
