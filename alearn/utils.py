import streamlit as st


def check_column_types():
    return list(set([st.session_state['col_type'][col]
              for col in st.session_state['col_status']
              if st.session_state['col_status'][col]]))


def check_alearn_loop():
    if 'alearn_loop' not in st.session_state:
        st.session_state['alearn_loop'] = {
            'step': 0,
            'seed_annotation': {},
        }


def next_alearn_step():
    st.session_state['alearn_loop']['step'] += 1