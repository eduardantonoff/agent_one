import streamlit as st

if 'embed' in st.session_state:
    texts = [doc['text'] for doc in st.session_state.embed.store.values()]
    st.write(texts)
else:
    st.write([])