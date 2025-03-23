# app.py
import streamlit as st
from ats_model import ATSModel, load_model, save_model
import pdfplumber


def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text


def main():
    try:
        ats_model = load_model()
    except:
        ats_model = ATSModel()
        save_model(ats_model)

    st.title("ATS Resume Matcher")
    st.markdown("Upload your resume (PDF) and compare it against a job description!")

    job_desc = st.text_area("Paste Job Description Here", height=200)
    resume_file = st.file_uploader("Upload Resume (PDF)", type="pdf")

    if st.button("Analyze"):
        if job_desc and resume_file:
            resume_text = extract_text_from_pdf(resume_file)
            if not resume_text.strip():
                st.error("Could not extract text from the PDF. Please ensure it contains readable content.")
                return

            similarity_score = ats_model.calculate_similarity(job_desc, resume_text)
            percentage_score = similarity_score * 100

            st.subheader("Results")
            st.write(f"**Match Percentage:** {percentage_score:.2f}%")
            progress_value = int(max(0, min(100, percentage_score)))
            st.progress(progress_value)

            missing_keywords = ats_model.find_missing_keywords(job_desc, resume_text)

            st.subheader("Missing Technical Keywords")
            if missing_keywords:
                for keyword in sorted(missing_keywords):
                    st.write(f"- {keyword.title()}")
            else:
                st.success("No significant technical keywords missing!")

            st.subheader("Recommendations")
            recommendations = ats_model.generate_recommendations(missing_keywords)
            st.markdown(recommendations)

            save_model(ats_model)
        else:
            st.error("Please provide both a job description and a resume PDF!")


if __name__ == "__main__":
    main()