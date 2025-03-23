import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
import re

# Load spaCy model globally
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

class ATSModel:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 3))

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text

    def extract_keywords(self, text, top_n=10):
        text = self.preprocess_text(text)
        doc = nlp(text)

        candidates = []
        for chunk in doc.noun_chunks:
            chunk_text = chunk.text.lower().strip()
            if len(chunk_text.split()) <= 3:
                candidates.append(chunk_text)

        for token in doc:
            if token.pos_ == 'PROPN' and len(token.text) > 2:
                candidates.append(token.text.lower())

        if not candidates:
            return []

        tfidf_matrix = self.tfidf_vectorizer.fit_transform([text] + candidates)
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_matrix[0].toarray()[0]

        candidate_scores = {}
        for candidate in candidates:
            candidate_words = candidate.split()
            try:
                score = sum(tfidf_scores[np.where(feature_names == word)[0][0]]
                            for word in candidate_words if word in feature_names) / len(candidate_words)
                candidate_scores[candidate] = score
            except:
                continue

        generic_terms = {'experience', 'skills', 'knowledge', 'development', 'technical', 'proficiency',
                         'career', 'basic', 'handson', 'fundamentals', 'projects', 'professionals',
                         'innovative', 'engineering'}

        final_keywords = []
        for candidate, score in sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True):
            if any(word in generic_terms for word in candidate.split()):
                continue
            if any(token.pos_ == 'PROPN' or token.text.isupper() for token in nlp(candidate)):
                score *= 1.5
            final_keywords.append((candidate, score))

        return [kw for kw, score in sorted(final_keywords, key=lambda x: x[1], reverse=True)][:top_n]

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
