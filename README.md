# AI Resume Analyzer 📝

An AI-powered web application that analyzes resumes against job descriptions to simulate ATS scoring and provide actionable feedback using NLP and LLMs.

## 🚀 Live Demo
🔗 https://resume-analyzer.streamlit.app

## 🔍 Features
- PDF resume text extraction
- Job description comparison
- ATS-style similarity scoring using Sentence Transformers
- AI-generated evaluation and improvement suggestions using Groq LLM
- Downloadable analysis report

## 🛠️ Tech Stack
- Python
- Streamlit
- Sentence Transformers (all-mpnet-base-v2)
- Groq LLM (LLaMA-based)
- Scikit-learn
- PDFMiner

## 📦 Installation (Local)
```bash
git clone https://github.com/reeti223/resume-analyzer.git
cd resume-analyzer
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run main.py
