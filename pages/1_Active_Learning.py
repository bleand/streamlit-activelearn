import streamlit as st
from alearn.binary import nlp_binary
st.set_page_config(layout="wide")

print('Step', st.session_state['alearn_loop']['step'])

#with st.expander('Session State'):
    #st.json(st.session_state)

if st.session_state['model_type'] == 'Binary':
    nlp_binary()