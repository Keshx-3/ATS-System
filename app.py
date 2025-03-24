import streamlit as st
import pdfplumber
from ats_model import ATSModel

# Initialize ATS Model
ats_model = ATSModel()

# Streamlit UI
st.set_page_config(page_title="ATS Resume Matcher", layout="centered")
st.title("📄 ATS Resume Matcher")
st.markdown("Upload your resume and job description to check how well they match. Get keyword suggestions to improve your resume score!")

# Upload Job Description
job_desc = st.text_area("📋 Paste Job Description here", height=250)

# Upload Resume PDF
resume_file = st.file_uploader("📎 Upload your Resume (PDF format only)", type=['pdf'])

# Process on button click
if st.button("🔍 Analyze Match"):
    if not job_desc or not resume_file:
        st.warning("Please provide both the Job Description and Resume PDF.")
    else:
        try:
            # Extract text from resume PDF
            with pdfplumber.open(resume_file) as pdf:
                resume_text = ''
                for page in pdf.pages:
                    resume_text += page.extract_text() or ''

            if not resume_text.strip():
                st.error("Could not extract text from PDF. Please try another file.")
            else:
                # Calculate similarity
                similarity_score = ats_model.calculate_similarity(job_desc, resume_text)
                st.markdown(f"### ✅ Similarity Score: **{similarity_score * 100:.2f}%**")

                # Extract and show missing keywords
                missing_keywords = ats_model.find_missing_keywords(job_desc, resume_text)
                recommendations = ats_model.generate_recommendations(missing_keywords)

                st.markdown("### 📌 Missing Keywords / Suggestions")
                st.text(recommendations)

        except Exception as e:
            st.error(f"Something went wrong during processing: {e}")
