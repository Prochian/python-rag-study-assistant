import json, time
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
docs = json.loads((ROOT/'data/corpus.json').read_text(encoding='utf-8'))
qs = json.loads((ROOT/'data/evaluation.json').read_text(encoding='utf-8'))
texts = [d['title']+' '+d['text'] for d in docs]
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3,5), sublinear_tf=True)
start=time.perf_counter(); X=vectorizer.fit_transform(texts); Q=vectorizer.transform([q['query'] for q in qs]); sims=cosine_similarity(Q,X); elapsed=time.perf_counter()-start
r1=r3=0; rr=0; ranks=[]
for row,q in zip(sims,qs):
    order=row.argsort()[::-1]
    target=next(i for i,d in enumerate(docs) if d['id']==q['relevant'])
    rank=list(order).index(target)+1; ranks.append(rank)
    r1 += rank<=1; r3 += rank<=3; rr += 1/rank
n=len(qs)
metrics={
 'documents':len(docs),'queries':n,
 'recall_at_1':round(r1/n,4),'recall_at_3':round(r3/n,4),
 'mrr':round(rr/n,4),'mean_ms':round(elapsed/n*1000,3),
 'min_rank':min(ranks),'max_rank':max(ranks)
}
(ROOT/'results.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
print(json.dumps(metrics,indent=2))
