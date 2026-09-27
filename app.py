import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))

import streamlit as st
from rag import LocalRAG

st.set_page_config(page_title='Python RAG Study Assistant', page_icon='📚')
st.title('Python RAG Study Assistant')
st.write('A small study assistant built from Python documentation.')

@st.cache_resource
def load_rag():
    return LocalRAG()

rag = load_rag()

with st.sidebar:
    top_k = st.slider('Number of sources', 1, 5, 3)
    st.caption(f'Retriever: {rag.model_name}')
    st.caption('Local demo: Ollama + Llama 3.2. Online demo: evidence-based fallback if Ollama is not available.')

question = st.text_input('Ask a Python question', 'What is a list comprehension?')

if question:
    results, ms = rag.retrieve(question, top_k)
    st.subheader('Answer')
    try:
        answer = rag.answer_with_ollama(question, results)
        st.write(answer)
    except Exception:
        st.write(rag.simple_answer(question, results))
        st.caption('Online demo mode: the answer is taken from the retrieved course material because no Ollama server is available.')

    st.subheader('Sources')
    for i, r in enumerate(results, 1):
        with st.expander(f'{i}. {r["title"]} — score {r["score"]:.3f}'):
            st.write(r['text'])
            st.caption(r['url'])
    st.caption(f'Retrieval time: {ms:.2f} ms')
