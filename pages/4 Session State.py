import streamlit as st

if 'abot' in st.session_state:
    state = st.session_state.abot.graph.get_state(st.session_state.thread).values
    if 'messages' in state:
        messages = state["messages"]
        st.write(messages)
    else:
        st.write([])
else:
    st.write([])