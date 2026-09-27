# Python RAG Study Assistant

**Student:** Emre Demirtas  
**Student ID:** 57229

This is a small Retrieval-Augmented Generation project based on Python documentation.

## What it does

1. Stores a small collection of Python documentation.
2. Finds the most relevant documents for a question.
3. Uses a local Ollama model when it is available.
4. If Ollama is not available, the browser demo shows the retrieved source text instead of making up an answer.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

For the full local LLM version, install Ollama and run:

```bash
ollama pull llama3.2:3b
```

Then open the Streamlit page.

## Dataset

The included `data/corpus.json` is a small fixture made from Python documentation. `src/scrape_python_docs.py` can be used to collect fresh pages from the official Python documentation.

## Evaluation

Run:

```bash
python src/evaluate.py
```

The fixed benchmark contains 32 questions and measures Recall@1, Recall@3 and MRR.

## Online demo

https://python-rag-study-assistant-emre.streamlit.app/
