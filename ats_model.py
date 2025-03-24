import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from keybert import KeyBERT
import re

class ATSModel:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.keyword_extractor = KeyBERT(model=self.model)

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text

    def extract_keywords(self, text, top_n=15):
        text = self.preprocess_text(text)

        # Extract keywords using KeyBERT
        keywords = self.keyword_extractor.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 2),
            stop_words='english',
            use_maxsum=True,
            nr_candidates=30,
            top_n=top_n
        )

        return [kw[0].lower() for kw in keywords]

    def get_embedding(self, text):
        return self.model.encode(text)

    def calculate_similarity(self, job_desc, resume):
        job_embedding = self.get_embedding(job_desc)
        resume_embedding = self.get_embedding(resume)
        similarity = cosine_similarity([job_embedding], [resume_embedding])[0][0]
        return similarity

    def find_missing_keywords(self, job_desc, resume):
        job_keywords = set(self.extract_keywords(job_desc))
        resume_keywords = set(self.extract_keywords(resume))
        missing_keywords = job_keywords - resume_keywords
        return missing_keywords

    def generate_recommendations(self, missing_keywords):
        if not missing_keywords:
            return "Your resume aligns well with the job description's technical requirements!"
        recommendations = ["Consider including the following technical skills in your resume:"]
        recommendations.extend([f"- {keyword.title()}" for keyword in sorted(missing_keywords)])
        return "\n".join(recommendations)

def save_model(model, filename='ats_model.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

def load_model(filename='ats_model.pkl'):
    with open(filename, 'rb') as f:
        return pickle.load(f)

if __name__ == "__main__":
    try:
        ats_model = ATSModel()
        save_model(ats_model)
        print("Model initialized and saved successfully!")
    except Exception as e:
        print(f"Error initializing model: {e}")
