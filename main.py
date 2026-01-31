import streamlit as st
from pdfminer.high_level import extract_text
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
import re
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# App title
st.title("AI Resume Analyzer 📝")

# Session state
if "submitted" not in st.session_state:
    st.session_state.submitted = False
if "resume_text" not in st.session_state:
    st.session_state.resume_text = ""
if "job_desc" not in st.session_state:
    st.session_state.job_desc = ""

# Extract text from PDF
def extract_pdf_text(file):
    try:
        return extract_text(file)
    except Exception as e:
        return str(e)

# ATS similarity score
def calculate_similarity(resume, jd):
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    emb1 = model.encode([resume])
    emb2 = model.encode([jd])
    return cosine_similarity(emb1, emb2)[0][0]

# LLM analysis
def get_report(resume, jd):
    client = Groq(api_key=api_key)

    prompt = f"""
    You are an AI Resume Analyzer.
    Analyze the resume against the job description.
    Give scores out of 5 with emojis (✅ ❌ ⚠️).
    End with: Suggestions to improve your resume.

    Resume:
    {resume}

    Job Description:
    {jd}
    """

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content

# Extract scores from report
def extract_scores(text):
    matches = re.findall(r"(\d+(?:\.\d+)?)/5", text)
    return [float(m) for m in matches]

# UI Form
if not st.session_state.submitted:
    with st.form("resume_form"):
        resume_file = st.file_uploader("Upload Resume (PDF)", type="pdf")
        st.session_state.job_desc = st.text_area("Paste Job Description")
        submitted = st.form_submit_button("Analyze")

        if submitted and resume_file and st.session_state.job_desc:
            st.session_state.resume_text = extract_pdf_text(resume_file)
            st.session_state.submitted = True
            st.rerun()

# Show results
if st.session_state.submitted:
    st.info("Analyzing resume...")

    ats_score = calculate_similarity(
        st.session_state.resume_text, st.session_state.job_desc
    )

    col1, col2 = st.columns(2)
    col1.metric("ATS Similarity Score", round(ats_score, 3))

    report = get_report(
        st.session_state.resume_text, st.session_state.job_desc
    )

    scores = extract_scores(report)
    avg_score = round(sum(scores) / (5 * len(scores)), 2) if scores else 0
    col2.metric("AI Average Score", avg_score)

    st.subheader("AI Generated Report")
    st.markdown(report)

    st.download_button(
        "Download Report",
        report,
        file_name="resume_report.txt",
    )

