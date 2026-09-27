import json, re, os, time
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'

class LocalRAG:
    def __init__(self, corpus_path=DATA/'corpus.json'):
        self.docs = json.loads(Path(corpus_path).read_text(encoding='utf-8'))
        self.model_name = 'tfidf-char-fallback'
        self.vectorizer = None
        self.matrix = None
        self.semantic = False
        try:
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer(os.getenv('EMBEDDING_MODEL','sentence-transformers/all-MiniLM-L6-v2'))
            texts = [d['title'] + '\n' + d['text'] for d in self.docs]
            self.matrix = self.encoder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            self.model_name = os.getenv('EMBEDDING_MODEL','sentence-transformers/all-MiniLM-L6-v2')
            self.semantic = True
        except Exception:
            self.vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3,5), sublinear_tf=True)
            self.matrix = self.vectorizer.fit_transform([d['title'] + '\n' + d['text'] for d in self.docs])

    def retrieve(self, query, k=4):
        start = time.perf_counter()
        if self.semantic:
            q = self.encoder.encode([query], normalize_embeddings=True)
            scores = np.asarray(q @ np.asarray(self.matrix).T)[0]
        else:
            q = self.vectorizer.transform([query])
            scores = cosine_similarity(q, self.matrix)[0]
        idx = np.argsort(scores)[::-1][:k]
        elapsed = (time.perf_counter() - start) * 1000
        return [{**self.docs[i], 'score': float(scores[i])} for i in idx], elapsed

    def build_prompt(self, query, results):
        context = '\n\n'.join([f"SOURCE {i+1}: {r['title']}\n{r['text']}" for i,r in enumerate(results)])
        return f'''You are a Python study assistant. Answer only from the supplied sources. If the sources do not contain the answer, say: "I could not find that in the course material." Keep the answer concise and explain the key idea. Mention the source title in parentheses when useful.\n\nCOURSE MATERIAL:\n{context}\n\nQUESTION: {query}\nANSWER:'''

    def answer_with_ollama(self, query, results, model='llama3.2:3b'):
        import requests
        prompt = self.build_prompt(query, results)
        r = requests.post('http://localhost:11434/api/generate', json={
            'model': model, 'prompt': prompt, 'stream': False,
            'options': {'temperature': 0.1}
        }, timeout=120)
        r.raise_for_status()
        return r.json()['response']

    def simple_answer(self, query, results):
        # Used by the browser demo when Ollama is not available.
        # It returns the most relevant source passage rather than inventing an answer.
        if not results:
            return "I could not find that in the course material."
        best = results[0]
        return f"According to the retrieved Python documentation ({best['title']}):\n\n{best['text']}"
