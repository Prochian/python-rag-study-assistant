import json, re, time
from pathlib import Path
import requests
from bs4 import BeautifulSoup

OUT = Path(__file__).resolve().parents[1] / 'data' / 'scraped_docs.json'
URLS = [
 'https://docs.python.org/3/tutorial/introduction.html',
 'https://docs.python.org/3/tutorial/controlflow.html',
 'https://docs.python.org/3/tutorial/datastructures.html',
 'https://docs.python.org/3/tutorial/errors.html',
 'https://docs.python.org/3/library/pathlib.html',
 'https://docs.python.org/3/library/json.html',
 'https://docs.python.org/3/library/re.html',
 'https://docs.python.org/3/library/datetime.html',
 'https://docs.python.org/3/library/collections.html',
]

def clean(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk(text, size=1100, overlap=180):
    words=text.split(); out=[]; start=0
    while start < len(words):
        end=min(len(words), start+size)
        out.append(' '.join(words[start:end]))
        if end == len(words): break
        start=end-overlap
    return out

rows=[]
for url in URLS:
    r=requests.get(url,timeout=30,headers={'User-Agent':'University-RAG-Project/1.0'})
    r.raise_for_status()
    soup=BeautifulSoup(r.text,'html.parser')
    main=soup.find('main') or soup
    for tag in main(['script','style','nav']): tag.decompose()
    title=clean(main.find('h1').get_text(' ',strip=True)) if main.find('h1') else url
    text=clean(main.get_text(' ',strip=True))
    for j,c in enumerate(chunk(text)):
        rows.append({'id':f'{url.split("/")[-1]}_{j}','title':title,'url':url,'text':c})
    time.sleep(0.2)
OUT.write_text(json.dumps(rows,indent=2,ensure_ascii=False),encoding='utf-8')
print(f'Wrote {len(rows)} chunks to {OUT}')
