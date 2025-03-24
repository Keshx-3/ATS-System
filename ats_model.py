from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
from sklearn.metrics.pairwise import cosine_similarity
import re

class ATSModel:
    def __init__(self):
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.keyword_extractor = KeyBERT(model=self.embedding_model)
        except Exception as e:
            print(f"Model init error: {e}")
            self.embedding_model = None
            self.keyword_extractor = None

    def preprocess_text(self, text):
        return re.sub(r'[^\w\s]', '', text.lower())

    def extract_keywords(self, text, top_n=10):
        if self.keyword_extractor is None:
            return []
        text = self.preprocess_text(text)
        try:
            keywords = self.keyword_extractor.extract_keywords(
                text,
                keyphrase_ngram_range=(1, 2),
                stop_words='english',
                use_maxsum=True,
                nr_candidates=30,
                top_n=top_n
            )
            return [kw[0].lower() for kw in keywords]
        except Exception as e:
            print(f"Keyword extraction error: {e}")
            return []

    def get_embedding(self, text):
        if self.embedding_model:
            return self.embedding_model.encode(text)
        return []

    def calculate_similarity(self, job_desc, resume):
        try:
            job_emb = self.get_embedding(job_desc)
            resume_emb = self.get_embedding(resume)
            return cosine_similarity([job_emb], [resume_emb])[0][0]
        except Exception as e:
            print(f"Similarity error: {e}")
            return 0.0

    def find_missing_keywords(self, job_desc, resume):
        job_kw = set(self.extract_keywords(job_desc))
        resume_kw = set(self.extract_keywords(resume))
        return job_kw - resume_kw

    def generate_recommendations(self, missing_keywords):
        if not missing_keywords:
            return "🎉 Your resume aligns well with the job description's key terms!"
        rec = ["💡 Consider including these missing keywords:"]
        rec += [f"- {kw.title()}" for kw in sorted(missing_keywords)]
        return "\n".join(rec)
