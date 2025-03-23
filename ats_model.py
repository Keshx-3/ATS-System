# ats_model.py
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import re

class ATSModel:
    def __init__(self):
        self.model = None
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 3))
        
    def _load_model(self):
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        return self.model
    
    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text
    
    def extract_keywords(self, text, top_n=10):
        text = self.preprocess_text(text)
        
        # Use TF-IDF directly on the text
        tfidf_matrix = self.tfidf_vectorizer.fit_transform([text])
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_matrix.toarray()[0]
        
        # Get keyword scores
        keyword_scores = {feature: score for feature, score in zip(feature_names, tfidf_scores)}
        
        # Filter out generic terms
        generic_terms = {'experience', 'skills', 'knowledge', 'development', 'technical', 'proficiency', 
                         'career', 'basic', 'handson', 'fundamentals', 'projects', 'professionals', 
                         'innovative', 'engineering'}
        
        final_keywords = []
        for keyword, score in sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True):
            if any(word in generic_terms for word in keyword.split()):
                continue
            final_keywords.append(keyword)
            if len(final_keywords) >= top_n:
                break
        
        return final_keywords
    
    def get_embedding(self, text):
        model = self._load_model()
        return model.encode(text)
    
    def calculate_similarity(self, job_desc, resume):
        model = self._load_model()
        job_embedding = model.encode(job_desc)
        resume_embedding = model.encode(resume)
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
