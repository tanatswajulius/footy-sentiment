import json
import numpy as np
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer

try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

class Analyzer:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def score_sentiment(self, text: str) -> float:
        return float(self.sia.polarity_scores(text)["compound"])

    def embed(self, texts):
        vecs = self.model.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
        return vecs.tolist(), int(vecs.shape[1]) if hasattr(vecs, "shape") else len(vecs[0])
