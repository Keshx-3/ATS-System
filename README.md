# 📄 ATS Resume Matcher System

An AI-powered Applicant Tracking System (ATS) that analyzes job descriptions and resumes to calculate ATS scores, identify missing keywords, and provide actionable feedback to improve resume matching against job roles. This tool assists job seekers in optimizing their resumes to pass through ATS filters used by modern recruitment systems.

---

## 🚀 Features

- 🔍 **ATS Resume Matching Score**  
  Calculates similarity score between job descriptions and resumes using NLP techniques.

- 📊 **Missing Keyword Analysis**  
  Highlights relevant keywords from the job description that are absent in the resume.

- 🧠 **Keyword Extraction using KeyBERT**  
  Leverages KeyBERT and transformer models to extract meaningful keywords from job descriptions and resumes.

- 📄 **PDF Resume Parsing**  
  Supports uploading resumes in `.pdf` format and extracts raw text using `pdfplumber`.

- 📈 **Visual Feedback Dashboard (via Streamlit)**  
  Interactive UI for job seekers to view ATS scores and keyword gaps.

---

## 🧠 Tech Stack

| Component         | Technology                         |
|------------------|------------------------------------|
| Frontend UI       | Streamlit                          |
| NLP Engine        | Sentence Transformers, KeyBERT    |
| Resume Parsing    | pdfplumber                         |
| Machine Learning  | scikit-learn, NumPy                |
| Keyword Analysis  | KeyBERT, Transformers              |
| Visualization     | Streamlit Widgets & Charts         |

---

## 🧪 How It Works

1. **Upload Resume** (`.pdf` format)
2. **Paste Job Description**
3. **Get Results**:
   - ATS Match Score (0–100%)
   - Missing Keywords
   - Recommendations for improvement

---

## 📷 Screenshots

![image](https://github.com/user-attachments/assets/deab1a9c-82c5-44c0-aea3-a583f058c5dc)
---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/ats-resume-matcher.git
cd ats-resume-matcher








