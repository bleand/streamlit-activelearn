import streamlit as st
from alearn.controller_builder import controller_from_settings
st.set_page_config(layout="wide")

# print('Step', st.session_state['alearn_loop']['step'])


if 'controller' not in st.session_state:
    st.session_state['controller'] = controller_from_settings(st.session_state)

st.session_state['controller'].run()
