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

# A few clear Python terms help prevent an unrelated question from being
# presented with an unrelated documentation passage in the online demo.
PYTHON_TERMS = {
    'python', 'list', 'tuple', 'string', 'str', 'dict', 'dictionary', 'set',
    'loop', 'for', 'while', 'function', 'class', 'object', 'exception',
    'error', 'syntax', 'module', 'package', 'import', 'json', 'pathlib',
    'regex', 'regular expression', 'datetime', 'deque', 'counter',
    'comprehension', 'iterator', 'generator', 'file', 'files', 'path',
    'variable', 'lambda', 'inheritance', 'decorator', 'queue', 'stack'
}

question = st.text_input('Ask a Python question', 'What is a list comprehension?')

if question.strip():
    results, ms = rag.retrieve(question, top_k)
    q = question.lower()
    looks_python_related = any(term in q for term in PYTHON_TERMS)

    # The online demo uses retrieval only when there is a reasonable Python
    # topic match. This prevents unrelated questions from returning an
    # arbitrary Python paragraph.
    best_score = results[0]['score'] if results else 0.0
    relevant = bool(results) and looks_python_related and best_score >= 0.10

    st.subheader('Answer')
    if not relevant:
        st.write("I couldn't find an answer to this question in the Python documentation used by this project.")
    else:
        try:
            answer = rag.answer_with_ollama(question, results)
            st.write(answer)
        except Exception:
            st.write(
                f"According to the retrieved Python documentation ({results[0]['title']}):\n\n"
                f"{results[0]['text']}"
            )

    if relevant:
        st.subheader('Sources')
        for i, r in enumerate(results, 1):
            with st.expander(f'{i}. {r["title"]} — score {r["score"]:.3f}'):
                st.write(r['text'])
                st.caption(r['url'])

    st.caption(f'Retrieval time: {ms:.2f} ms')
